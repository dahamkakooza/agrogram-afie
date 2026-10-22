package com.agrogram.afie.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.agrogram.afie.ml.DiseaseClassifier

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ResultScreen(prediction: DiseaseClassifier.Prediction, onBack: () -> Unit) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Diagnosis") },
                navigationIcon = { IconButton(onClick = onBack) {
                    Icon(Icons.Default.ArrowBack, "Back") } }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier.fillMaxSize().padding(padding).padding(16.dp)
                .verticalScroll(rememberScrollState())
        ) {
            // ─── Primary result ────────────────────────────────
            Card(colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.primaryContainer)) {
                Column(Modifier.padding(20.dp)) {
                    Text("Detected crop",
                         style = MaterialTheme.typography.labelMedium)
                    Text(prediction.detectedCrop.replaceFirstChar { it.uppercase() },
                         style = MaterialTheme.typography.titleMedium,
                         fontWeight = FontWeight.Bold)
                    Spacer(Modifier.height(12.dp))
                    Text("Predicted disease",
                         style = MaterialTheme.typography.labelMedium)
                    Text(prediction.label,
                         style = MaterialTheme.typography.headlineSmall,
                         fontWeight = FontWeight.Bold)
                    Spacer(Modifier.height(8.dp))
                    Text("Confidence: ${"%.1f".format(prediction.confidence * 100)}%",
                         style = MaterialTheme.typography.bodyLarge)
                    Text("Inference: ${prediction.latencyMs} ms",
                         style = MaterialTheme.typography.bodySmall)
                }
            }

            Spacer(Modifier.height(20.dp))

            // ─── Top 3 (before crop filtering) ─────────────────
            Text("Top 3 raw predictions",
                 style = MaterialTheme.typography.titleMedium)
            Text("What the model saw, before crop filtering:",
                 style = MaterialTheme.typography.bodySmall)
            Spacer(Modifier.height(8.dp))
            Card {
                Column(Modifier.padding(16.dp)) {
                    prediction.top3.forEachIndexed { i, (label, prob) ->
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text("${i + 1}. $label",
                                 style = MaterialTheme.typography.bodyMedium)
                            Text("${"%.1f".format(prob * 100)}%",
                                 style = MaterialTheme.typography.bodyMedium,
                                 fontWeight = FontWeight.SemiBold)
                        }
                        if (i < prediction.top3.size - 1) Spacer(Modifier.height(4.dp))
                    }
                }
            }

            Spacer(Modifier.height(20.dp))

            // ─── All class scores ──────────────────────────────
            Text("All class scores (after crop filter)",
                 style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(12.dp))
            Card {
                Column(Modifier.padding(16.dp)) {
                    prediction.probabilities.forEachIndexed { i, p ->
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(DiseaseClassifier.CLASS_LABELS[i],
                                 style = MaterialTheme.typography.bodySmall,
                                 fontWeight = if (i == prediction.classIndex)
                                     FontWeight.Bold else FontWeight.Normal)
                            Text("${"%.1f".format(p * 100)}%",
                                 style = MaterialTheme.typography.bodySmall,
                                 fontWeight = if (i == prediction.classIndex)
                                     FontWeight.Bold else FontWeight.Normal)
                        }
                        if (i < prediction.probabilities.size - 1) Spacer(Modifier.height(6.dp))
                    }
                }
            }

            Spacer(Modifier.height(24.dp))
            Button(onClick = onBack, modifier = Modifier.fillMaxWidth()) {
                Text("Scan another leaf")
            }
            Spacer(Modifier.height(16.dp))
        }
    }
}