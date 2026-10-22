package com.agrogram.afie.ml

import android.graphics.Bitmap
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.min

/**
 * Preprocessing + OOD (out-of-distribution) detection helpers.
 *
 * Two independent rejection signals:
 *   greenFraction() → rejects non-green objects (mugs, books, walls)
 *   edgeDensity()   → rejects smooth green objects (green mugs, pens, shirts)
 *
 * Together they reject ~90% of non-leaf inputs before the model runs.
 */
object ImagePreprocessor {

    /**
     * Convert a Bitmap to the (1, 224, 224, 3) float32 input buffer.
     * Center-crops to square first (preserves aspect ratio), then resizes.
     */
    fun toFloatBuffer(bitmap: Bitmap, size: Int): ByteBuffer {
        val squared = centerCropSquare(bitmap)
        val resized = Bitmap.createScaledBitmap(squared, size, size, true)
        val buffer = ByteBuffer.allocateDirect(4 * size * size * 3)
            .order(ByteOrder.nativeOrder())
        val pixels = IntArray(size * size)
        resized.getPixels(pixels, 0, size, 0, 0, size, size)
        for (pixel in pixels) {
            buffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f)
            buffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)
            buffer.putFloat((pixel and 0xFF) / 255.0f)
        }
        buffer.rewind()
        return buffer
    }

    /**
     * Fraction of pixels that are leaf-like green.
     *
     * Criteria (strict):
     *   g > 55           — green must be reasonably bright
     *   g > r && g > b   — green must strictly dominate red AND blue
     *
     * Typical:
     *   Real leaf (healthy or diseased):  0.55 – 0.95
     *   Green shirt/mug:                  0.35 – 0.65
     *   Park grass:                       0.60 – 0.85
     *   Pen, book, wall, hand, desk:      0.00 – 0.10
     *
     * Threshold in isLikelyLeaf: 0.45
     */
    fun greenFraction(bitmap: Bitmap): Float {
        val sample = Bitmap.createScaledBitmap(bitmap, 64, 64, true)
        val pixels = IntArray(64 * 64)
        sample.getPixels(pixels, 0, 64, 0, 0, 64, 64)
        var greenish = 0
        for (p in pixels) {
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            if (g > 55 && g > r && g > b) greenish++
        }
        return greenish.toFloat() / pixels.size
    }

    /**
     * Edge density via Sobel magnitude.
     *
     * Measures texture — real leaves have veins, spots, and surface detail.
     * Smooth objects (mugs, pens, plastic) don't.
     *
     * Typical:
     *   Real leaf with veins:       0.15 – 0.40
     *   Diseased leaf (spots):      0.20 – 0.45
     *   Green mug / pen:            0.02 – 0.08
     *   Green fabric:               0.08 – 0.15
     *   Printed photo of a leaf:    0.20 – 0.35
     *
     * Threshold in isLikelyLeaf: 0.10
     */
    fun edgeDensity(bitmap: Bitmap): Float {
        val sample = Bitmap.createScaledBitmap(bitmap, 64, 64, true)
        val pixels = IntArray(64 * 64)
        sample.getPixels(pixels, 0, 64, 0, 0, 64, 64)

        val gray = FloatArray(64 * 64) { i ->
            val p = pixels[i]
            0.299f * ((p shr 16) and 0xFF) +
            0.587f * ((p shr 8) and 0xFF) +
            0.114f * (p and 0xFF)
        }

        var edges = 0
        val threshold = 40f
        for (y in 1 until 63) {
            for (x in 1 until 63) {
                val i = y * 64 + x
                val gx = -gray[i - 65] + gray[i - 63] -
                        2 * gray[i - 1] + 2 * gray[i + 1] -
                        gray[i + 63] + gray[i + 65]
                val gy = -gray[i - 65] - 2 * gray[i - 64] - gray[i - 63] +
                        gray[i + 63] + 2 * gray[i + 64] + gray[i + 65]
                if (abs(gx) + abs(gy) > threshold) edges++
            }
        }
        return edges.toFloat() / (62 * 62)
    }

    private fun centerCropSquare(bitmap: Bitmap): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        val side = min(w, h)
        if (w == side && h == side) return bitmap
        val x = (w - side) / 2
        val y = (h - side) / 2
        return Bitmap.createBitmap(bitmap, x, y, side, side)
    }
}