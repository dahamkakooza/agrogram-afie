package com.agrogram.afie.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.CloudDone
import androidx.compose.material.icons.filled.CloudOff
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.agrogram.afie.data.AppDatabase
import com.agrogram.afie.data.DiagnosisEntity
import com.agrogram.afie.worker.SyncWorker
import androidx.compose.runtime.collectAsState
import kotlinx.coroutines.launch
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HistoryScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val dao = remember { AppDatabase.getInstance(context).diagnosisDao() }

    val diagnoses by dao.observeAll().collectAsState(initial = emptyList())
    val unsynced by dao.observeUnsyncedCount().collectAsState(initial = 0)

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("History (${diagnoses.size})") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back")
                    }
                },
                actions = {
                    if (unsynced > 0) {
                        IconButton(onClick = { SyncWorker.syncNow(context) }) {
                            BadgedBox(badge = { Badge { Text("$unsynced") } }) {
                                Icon(Icons.Default.CloudOff, "Sync now")
                            }
                        }
                    }
                }
            )
        }
    ) { padding ->
        if (diagnoses.isEmpty()) {
            Box(
                modifier = Modifier.fillMaxSize().padding(padding),
                contentAlignment = Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("No diagnoses yet",
                         style = MaterialTheme.typography.titleMedium)
                    Spacer(Modifier.height(8.dp))
                    Text("Scan a leaf from the camera screen",
                         style = MaterialTheme.typography.bodyMedium)
                }
            }
        } else {
            LazyColumn(
                modifier = Modifier.fillMaxSize().padding(padding),
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                items(diagnoses, key = { it.id }) { diagnosis ->
                    DiagnosisCard(
                        diagnosis = diagnosis,
                        onDelete = { scope.launch { dao.delete(diagnosis.id) } }
                    )
                }
            }
        }
    }
}

@Composable
private fun DiagnosisCard(diagnosis: DiagnosisEntity, onDelete: () -> Unit) {
    Card(modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(12.dp)) {
        Row(modifier = Modifier.padding(12.dp),
            verticalAlignment = Alignment.CenterVertically) {

            val file = File(diagnosis.imagePath)
            if (file.exists()) {
                AsyncImage(
                    model = file,
                    contentDescription = null,
                    modifier = Modifier.size(80.dp).clip(RoundedCornerShape(8.dp)),
                    contentScale = ContentScale.Crop
                )
            } else {
                Box(modifier = Modifier.size(80.dp).clip(RoundedCornerShape(8.dp)),
                    contentAlignment = Alignment.Center) {
                    Text("?", style = MaterialTheme.typography.headlineMedium)
                }
            }

            Spacer(Modifier.width(12.dp))

            Column(modifier = Modifier.weight(1f)) {
                Text(diagnosis.predictedLabel,
                     style = MaterialTheme.typography.titleSmall,
                     fontWeight = FontWeight.Bold)
                Text("Crop: ${diagnosis.cropType}",
                     style = MaterialTheme.typography.bodySmall)
                Text("Confidence: ${(diagnosis.confidence * 100).toInt()}%",
                     style = MaterialTheme.typography.bodySmall)
                Text(
                    SimpleDateFormat("MMM d, HH:mm", Locale.US)
                        .format(Date(diagnosis.createdAt)),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Spacer(Modifier.height(4.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        if (diagnosis.synced) Icons.Default.CloudDone
                        else Icons.Default.CloudOff,
                        contentDescription = null,
                        modifier = Modifier.size(14.dp),
                        tint = if (diagnosis.synced) MaterialTheme.colorScheme.primary
                               else MaterialTheme.colorScheme.error
                    )
                    Spacer(Modifier.width(4.dp))
                    Text(
                        if (diagnosis.synced) "Synced" else "Not synced",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            IconButton(onClick = onDelete) {
                Icon(Icons.Default.Delete, "Delete")
            }
        }
    }
}