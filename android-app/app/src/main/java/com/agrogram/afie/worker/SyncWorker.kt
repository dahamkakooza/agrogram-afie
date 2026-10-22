package com.agrogram.afie.worker

import android.content.Context
import android.util.Log
import androidx.work.*
import com.agrogram.afie.data.DiagnosisRepository
import com.agrogram.afie.network.RetrofitClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.concurrent.TimeUnit

class SyncWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        val repo = DiagnosisRepository(applicationContext)
        try {
            val unsynced = repo.getUnsynced()
            if (unsynced.isEmpty()) {
                Log.d(TAG, "Nothing to sync")
                return@withContext Result.success()
            }
            Log.d(TAG, "Syncing ${unsynced.size} diagnoses...")
            val items = unsynced.map { repo.toSyncItem(it) }
            val response = RetrofitClient.api.syncDiagnoses(items)
            Log.d(TAG, "Sync success: ${response.synced}")
            repo.markSynced(response.ids)
            Result.success()
        } catch (e: Exception) {
            Log.w(TAG, "Sync failed: ${e.message}")
            if (runAttemptCount >= 5) Result.failure() else Result.retry()
        }
    }

    companion object {
        private const val TAG = "SyncWorker"
        private const val WORK_NAME = "diagnosis_sync"

        fun schedule(context: Context) {
            val request = PeriodicWorkRequestBuilder<SyncWorker>(
                15, TimeUnit.MINUTES
            )
                .setConstraints(
                    Constraints.Builder()
                        .setRequiredNetworkType(NetworkType.CONNECTED)
                        .build()
                )
                .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 1, TimeUnit.MINUTES)
                .build()

            WorkManager.getInstance(context).enqueueUniquePeriodicWork(
                WORK_NAME, ExistingPeriodicWorkPolicy.KEEP, request)
            Log.d(TAG, "Periodic sync scheduled")
        }

        fun syncNow(context: Context) {
            val request = OneTimeWorkRequestBuilder<SyncWorker>()
                .setConstraints(
                    Constraints.Builder()
                        .setRequiredNetworkType(NetworkType.CONNECTED)
                        .build()
                )
                .build()
            WorkManager.getInstance(context).enqueue(request)
            Log.d(TAG, "One-time sync enqueued")
        }
    }
}
