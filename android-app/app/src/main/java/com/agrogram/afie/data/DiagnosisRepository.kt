package com.agrogram.afie.data

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.agrogram.afie.ml.DiseaseClassifier
import com.agrogram.afie.network.DiagnosisSyncItem
import kotlinx.coroutines.flow.Flow
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class DiagnosisRepository(private val context: Context) {

    private val dao = AppDatabase.getInstance(context).diagnosisDao()
    private val userId = UserPrefs.getOrCreateUserId(context)

    val allDiagnoses: Flow<List<DiagnosisEntity>> = dao.observeAll()
    val unsyncedCount: Flow<Int> = dao.observeUnsyncedCount()
    val totalCount: Flow<Int> = dao.observeTotalCount()

    companion object {
        private const val TAG = "DiagnosisRepository"
        private val CONTEXT_KEYS = listOf("crop_type", "season", "symptoms", "location")
    }

    suspend fun save(
        bitmap: Bitmap,
        prediction: DiseaseClassifier.Prediction,
        cropType: String
    ): String {
        val entityId = java.util.UUID.randomUUID().toString()

        val imageDir = File(context.filesDir, "diagnoses").apply { mkdirs() }
        val imageFile = File(imageDir, "$entityId.jpg")
        try {
            FileOutputStream(imageFile).use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 85, out)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save image: ${e.message}")
        }

        val attentionJson = JSONObject().apply {
            prediction.attentionWeights.forEachIndexed { i, v ->
                put(CONTEXT_KEYS.getOrElse(i) { "feature_$i" }, v.toDouble())
            }
        }

        val cropIndex = DiseaseClassifier.CROP_INDEX[cropType.lowercase()] ?: 3
        val contextArray = JSONArray().apply {
            for (i in 0 until DiseaseClassifier.CONTEXT_DIM) {
                put(if (i == cropIndex) 1.0 else 0.0)
            }
        }

        val entity = DiagnosisEntity(
            id = entityId,
            userId = userId,
            farmId = null,
            imagePath = imageFile.absolutePath,
            cropType = cropType.lowercase(),
            contextVector = contextArray.toString(),
            predictedClass = prediction.classIndex,
            predictedLabel = prediction.label,
            confidence = prediction.confidence,
            attentionWeights = attentionJson.toString(),
            modelVersion = "1.0.0",
            inferenceLatencyMs = prediction.latencyMs.toFloat(),
            createdAt = System.currentTimeMillis(),
            synced = false
        )

        dao.insert(entity)
        Log.d(TAG, "Saved ${entity.id} (${entity.predictedLabel})")
        return entityId
    }

    suspend fun getUnsynced(): List<DiagnosisEntity> = dao.getUnsynced()

    suspend fun markSynced(ids: List<String>) {
        val now = System.currentTimeMillis()
        for (id in ids) dao.markSynced(id, now)
        Log.d(TAG, "Marked ${ids.size} as synced")
    }

    suspend fun delete(id: String) = dao.delete(id)
    suspend fun deleteAll() = dao.deleteAll()

    fun toSyncItem(entity: DiagnosisEntity): DiagnosisSyncItem {
        val contextVector = try {
            val arr = JSONArray(entity.contextVector)
            List(arr.length()) { arr.getDouble(it).toFloat() }
        } catch (e: Exception) {
            listOf(0f, 0f, 0f, 1f)
        }

        val attentionWeights = try {
            val obj = JSONObject(entity.attentionWeights)
            obj.keys().asSequence().associateWith { obj.getDouble(it).toFloat() }
        } catch (e: Exception) {
            mapOf("crop_type" to 1.0f, "season" to 0.0f,
                  "symptoms" to 0.0f, "location" to 0.0f)
        }

        return DiagnosisSyncItem(
            userId = entity.userId,
            farmId = entity.farmId,
            imagePath = entity.imagePath,
            cropType = entity.cropType,
            contextVector = contextVector,
            predictedClass = entity.predictedClass,
            confidence = entity.confidence,
            attentionWeights = attentionWeights,
            modelVersion = entity.modelVersion,
            inferenceLatencyMs = entity.inferenceLatencyMs,
            deviceInfo = mapOf(
                "platform" to "android",
                "app_version" to "1.0",
                "model" to android.os.Build.MODEL
            ),
            createdAt = SimpleDateFormat(
                "yyyy-MM-dd'T'HH:mm:ss", Locale.US
            ).format(Date(entity.createdAt))
        )
    }
}
