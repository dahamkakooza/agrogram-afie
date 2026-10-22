package com.agrogram.afie.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface DiagnosisDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(diagnosis: DiagnosisEntity)

    @Query("SELECT * FROM diagnoses ORDER BY createdAt DESC")
    fun observeAll(): Flow<List<DiagnosisEntity>>

    @Query("SELECT * FROM diagnoses WHERE synced = 0 ORDER BY createdAt ASC")
    suspend fun getUnsynced(): List<DiagnosisEntity>

    @Query("UPDATE diagnoses SET synced = 1, syncedAt = :ts WHERE id = :id")
    suspend fun markSynced(id: String, ts: Long)

    @Query("SELECT COUNT(*) FROM diagnoses WHERE synced = 0")
    fun observeUnsyncedCount(): Flow<Int>

    @Query("SELECT COUNT(*) FROM diagnoses")
    fun observeTotalCount(): Flow<Int>

    @Query("SELECT * FROM diagnoses WHERE id = :id")
    suspend fun getById(id: String): DiagnosisEntity?

    @Query("DELETE FROM diagnoses WHERE id = :id")
    suspend fun delete(id: String)

    @Query("DELETE FROM diagnoses")
    suspend fun deleteAll()
}
