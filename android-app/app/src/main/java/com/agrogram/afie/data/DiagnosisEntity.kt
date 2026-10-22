package com.agrogram.afie.data

import androidx.room.Entity
import androidx.room.PrimaryKey
import java.util.UUID

@Entity(tableName = "diagnoses")
data class DiagnosisEntity(
    @PrimaryKey val id: String = UUID.randomUUID().toString(),
    val userId: String,
    val farmId: String? = null,
    val imagePath: String,
    val cropType: String,
    val contextVector: String,
    val predictedClass: Int,
    val predictedLabel: String,
    val confidence: Float,
    val attentionWeights: String,
    val modelVersion: String,
    val inferenceLatencyMs: Float,
    val createdAt: Long,
    val synced: Boolean = false,
    val syncedAt: Long? = null
)
