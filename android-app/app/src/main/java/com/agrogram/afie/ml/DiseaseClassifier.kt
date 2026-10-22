package com.agrogram.afie.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.Serializable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.min

class DiseaseClassifier(private val context: Context) {

    companion object {
        private const val TAG = "DiseaseClassifier"
        const val IMG_SIZE = 224
        const val NUM_CLASSES = 10
        const val CONTEXT_DIM = 4

        val CLASS_LABELS = listOf(
            "Maize Northern Leaf Blight",   // 0
            "Maize Common Rust",            // 1
            "Maize Healthy",                // 2
            "Tomato Early Blight",          // 3
            "Tomato Late Blight",           // 4
            "Tomato Leaf Mold",             // 5
            "Tomato Healthy",               // 6
            "Potato Early Blight",          // 7
            "Potato Late Blight",           // 8
            "Potato Healthy"                // 9
        )

        val CROP_INDEX = mapOf(
            "maize" to 0, "tomato" to 1, "potato" to 2, "other" to 3
        )

        val CROP_TO_CLASSES = mapOf(
            "maize" to listOf(0, 1, 2),
            "tomato" to listOf(3, 4, 5, 6),
            "potato" to listOf(7, 8, 9),
            "other" to (0..9).toList()
        )
    }

    data class Prediction(
        val classIndex: Int,
        val label: String,
        val confidence: Float,
        val probabilities: FloatArray,
        val attentionWeights: FloatArray,
        val latencyMs: Long,
        val rawMaxConfidence: Float,
        val cropType: String,
        val rawProbabilities: FloatArray,
        val detectedCrop: String,
        val cropConfidence: Float,
        val top3: List<Pair<String, Float>>
    ) : Serializable

    private val interpreter: Interpreter
    private val imgInputIndex: Int
    private val ctxInputIndex: Int
    private val probsOutputIndex: Int
    private val attnOutputIndex: Int

    init {
        val model = loadModelFile("agriyouthnet_context_plus_int8.tflite")
        interpreter = Interpreter(model, Interpreter.Options().apply { setNumThreads(4) })

        var imgIdx = -1; var ctxIdx = -1
        for (i in 0 until interpreter.inputTensorCount) {
            val shape = interpreter.getInputTensor(i).shape()
            if (shape.size == 4 && shape[3] == 3) imgIdx = i
            else if (shape.size == 2 && shape[1] == CONTEXT_DIM) ctxIdx = i
        }
        var probsIdx = -1; var attnIdx = -1
        for (i in 0 until interpreter.outputTensorCount) {
            val shape = interpreter.getOutputTensor(i).shape()
            if (shape.size == 2 && shape[1] == NUM_CLASSES) probsIdx = i
            else if (shape.size == 2 && shape[1] == CONTEXT_DIM) attnIdx = i
        }
        require(imgIdx >= 0 && ctxIdx >= 0) { "Could not locate model inputs" }
        require(probsIdx >= 0 && attnIdx >= 0) { "Could not locate model outputs" }
        imgInputIndex = imgIdx; ctxInputIndex = ctxIdx
        probsOutputIndex = probsIdx; attnOutputIndex = attnIdx

        Log.d(TAG, "Loaded: imgIdx=$imgIdx ctxIdx=$ctxIdx probsIdx=$probsIdx attnIdx=$attnIdx")
    }

    /**
     * OOD (out-of-distribution) detection: is this likely a leaf?
     *
     * Three checks, applied in order (cheapest first):
     *   1. Green fraction ≥ 0.45  → rejects non-green objects
     *   2. Edge density  ≥ 0.10  → rejects smooth green objects (mugs, pens)
     *   3. Multi-crop consensus  → rejects heterogeneous/ambiguous inputs
     *
     * Returns (isLikelyLeaf, reason). The `reason` is shown to the user
     * if the check fails.
     */
    fun isLikelyLeaf(bitmap: Bitmap): Pair<Boolean, String> {
        // ─── Check 1: Green fraction ────────────────────────────
        val green = ImagePreprocessor.greenFraction(bitmap)
        Log.d(TAG, "greenFraction=${"%.3f".format(green)}")
        if (green < 0.45f) {
            return Pair(false, "not green enough (${"%.0f".format(green * 100)}%)")
        }

        // ─── Check 2: Edge density (texture) ────────────────────
        val texture = ImagePreprocessor.edgeDensity(bitmap)
        Log.d(TAG, "edgeDensity=${"%.3f".format(texture)}")
        if (texture < 0.10f) {
            return Pair(false, "too smooth — not a leaf surface " +
                        "(${"%.0f".format(texture * 100)}% edge density)")
        }

        // ─── Check 3: Multi-crop consensus ──────────────────────
        val crops = listOf(
            centerCropPercent(bitmap, 0.95f, 0.5f, 0.5f),
            centerCropPercent(bitmap, 0.70f, 0.5f, 0.5f),
            centerCropPercent(bitmap, 0.80f, 0.5f, 0.35f),
            centerCropPercent(bitmap, 0.80f, 0.5f, 0.65f)
        )
        val detected = mutableListOf<String>()
        for (crop in crops) {
            val probs = runInference(
                ImagePreprocessor.toFloatBuffer(crop, IMG_SIZE), "other")
            val maize = probs[0] + probs[1] + probs[2]
            val tomato = probs[3] + probs[4] + probs[5] + probs[6]
            val potato = probs[7] + probs[8] + probs[9]
            val winner = when {
                maize >= tomato && maize >= potato -> "maize"
                tomato >= potato -> "tomato"
                else -> "potato"
            }
            detected.add(winner)
        }
        Log.d(TAG, "crop consensus: $detected")
        val counts = detected.groupingBy { it }.eachCount()
        val maxAgree = counts.values.maxOrNull() ?: 0
        if (maxAgree < 3) {
            return Pair(false, "inconsistent crop detection ($detected)")
        }
        return Pair(true, "OK — green ${"%.0f".format(green*100)}%, " +
                    "texture ${"%.0f".format(texture*100)}%, " +
                    "crop consensus $maxAgree/4")
    }

    /**
     * Run full inference with auto-detect (or override) + crop filtering.
     */
    fun predict(bitmap: Bitmap, cropTypeOverride: String? = null): Prediction {
        val start = System.currentTimeMillis()
        val imgBuffer = ImagePreprocessor.toFloatBuffer(bitmap, IMG_SIZE)

        // Pass 1: auto-detect via "other" context
        val pass1 = runInference(imgBuffer, "other")
        val pass1Log = pass1.mapIndexed { i, p -> "${i}:${"%.3f".format(p)}" }
            .joinToString(" ")
        Log.d(TAG, "PASS 1 raw: $pass1Log")

        val cropSums = mapOf(
            "maize" to (pass1[0] + pass1[1] + pass1[2]),
            "tomato" to (pass1[3] + pass1[4] + pass1[5] + pass1[6]),
            "potato" to (pass1[7] + pass1[8] + pass1[9])
        )
        val (detectedCrop, cropConf) = cropSums.maxByOrNull { it.value }!!
        Log.d(TAG, "detected crop=$detectedCrop (${"%.3f".format(cropConf)})")

        val usedCrop = cropTypeOverride?.lowercase()?.trim() ?: detectedCrop
        Log.d(TAG, "using crop for pass 2: $usedCrop")

        // Pass 2: classify within selected/detected crop
        val pass2 = if (usedCrop == "other") pass1
                    else runInference(imgBuffer, usedCrop)
        val pass2Log = pass2.mapIndexed { i, p -> "${i}:${"%.3f".format(p)}" }
            .joinToString(" ")
        Log.d(TAG, "PASS 2 raw: $pass2Log")

        // Filter to used crop's classes
        val allowed = CROP_TO_CLASSES[usedCrop] ?: (0..9).toList()
        val filtered = FloatArray(NUM_CLASSES)
        var sum = 0f
        for (c in allowed) { filtered[c] = pass2[c]; sum += pass2[c] }
        if (sum > 0f) for (c in allowed) filtered[c] /= sum

        val predIdx = filtered.indices.maxByOrNull { filtered[it] } ?: 0
        val rawMax = pass2.max()

        val top3 = pass1.indices.sortedByDescending { pass1[it] }.take(3)
            .map { CLASS_LABELS[it] to pass1[it] }

        val latency = System.currentTimeMillis() - start
        Log.d(TAG, "FINAL: $predIdx (${CLASS_LABELS[predIdx]}), " +
                    "conf=${"%.3f".format(filtered[predIdx])}, " +
                    "rawMax=${"%.3f".format(rawMax)}")

        return Prediction(
            classIndex = predIdx,
            label = CLASS_LABELS[predIdx],
            confidence = filtered[predIdx],
            probabilities = filtered,
            attentionWeights = FloatArray(CONTEXT_DIM),
            latencyMs = latency,
            rawMaxConfidence = rawMax,
            cropType = usedCrop,
            rawProbabilities = pass1.copyOf(),
            detectedCrop = detectedCrop,
            cropConfidence = cropConf,
            top3 = top3
        )
    }

    private fun runInference(imgBuffer: ByteBuffer, crop: String): FloatArray {
        val ctxBuffer = ByteBuffer.allocateDirect(4 * 4).order(ByteOrder.nativeOrder())
        val ctxArray = FloatArray(CONTEXT_DIM)
        ctxArray[CROP_INDEX[crop.lowercase()] ?: 3] = 1.0f
        ctxBuffer.asFloatBuffer().put(ctxArray)
        ctxBuffer.rewind()

        val probsOut = Array(1) { FloatArray(NUM_CLASSES) }
        val attnOut = Array(1) { FloatArray(CONTEXT_DIM) }
        val outputs = mutableMapOf<Int, Any>(
            probsOutputIndex to probsOut,
            attnOutputIndex to attnOut
        )
        val inputs = arrayOfNulls<Any>(interpreter.inputTensorCount)
        inputs[imgInputIndex] = imgBuffer
        inputs[ctxInputIndex] = ctxBuffer

        @Suppress("UNCHECKED_CAST")
        interpreter.runForMultipleInputsOutputs(inputs as Array<Any>, outputs)
        return probsOut[0]
    }

    private fun centerCropPercent(
        src: Bitmap, sizeFrac: Float, cx: Float, cy: Float
    ): Bitmap {
        val s = (min(src.width, src.height) * sizeFrac).toInt()
        val x = ((src.width * cx) - s / 2).toInt().coerceIn(0, src.width - s)
        val y = ((src.height * cy) - s / 2).toInt().coerceIn(0, src.height - s)
        return Bitmap.createBitmap(src, x, y, s, s)
    }

    fun close() = interpreter.close()

    private fun loadModelFile(name: String): MappedByteBuffer {
        val fd = context.assets.openFd(name)
        val input = FileInputStream(fd.fileDescriptor)
        return input.channel.map(
            FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }
}