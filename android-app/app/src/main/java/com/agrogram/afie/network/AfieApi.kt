package com.agrogram.afie.network

import com.google.gson.annotations.SerializedName
import retrofit2.http.Body
import retrofit2.http.POST

interface AfieApi {

    @POST("api/v1/diagnosis/sync")
    suspend fun syncDiagnoses(
        @Body items: List<DiagnosisSyncItem>
    ): SyncResponse
}

data class DiagnosisSyncItem(
    @SerializedName("user_id")              val userId: String,
    @SerializedName("farm_id")              val farmId: String?,
    @SerializedName("image_path")           val imagePath: String,
    @SerializedName("crop_type")            val cropType: String,
    @SerializedName("context_vector")       val contextVector: List<Float>,
    @SerializedName("predicted_class")      val predictedClass: Int,
    @SerializedName("confidence")           val confidence: Float,
    @SerializedName("attention_weights")    val attentionWeights: Map<String, Float>,
    @SerializedName("model_version")        val modelVersion: String,
    @SerializedName("inference_latency_ms") val inferenceLatencyMs: Float?,
    @SerializedName("device_info")          val deviceInfo: Map<String, Any>?,
    @SerializedName("created_at")           val createdAt: String
)

data class SyncResponse(
    @SerializedName("synced") val synced: Int,
    @SerializedName("ids")    val ids: List<String>
)
