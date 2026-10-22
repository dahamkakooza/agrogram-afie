package com.agrogram.afie.ui

import android.Manifest
import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.History
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.agrogram.afie.data.DiagnosisRepository
import com.agrogram.afie.ml.DiseaseClassifier
import com.agrogram.afie.worker.SyncWorker
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private const val MIN_RAW_CONFIDENCE = 0.65f

@OptIn(ExperimentalPermissionsApi::class, ExperimentalMaterial3Api::class)
@Composable
fun CameraScreen(
    onPredictionReady: (DiseaseClassifier.Prediction) -> Unit,
    onHistoryClick: () -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraPermission = rememberPermissionState(Manifest.permission.CAMERA)
    var cropOverride by remember { mutableStateOf<String?>(null) }
    var isProcessing by remember { mutableStateOf(false) }
    var errorMsg by remember { mutableStateOf<String?>(null) }
    val classifier = remember { DiseaseClassifier(context) }
    val repository = remember { DiagnosisRepository(context) }
    val scope = rememberCoroutineScope()

    val imageCapture = remember {
        ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .build()
    }
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
    val previewView = remember {
        PreviewView(context).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }
    }

    LaunchedEffect(cameraPermission.status) {
        if (cameraPermission.status.isGranted) {
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA,
                    preview, imageCapture)
            } catch (e: Exception) {
                errorMsg = "Camera init failed: ${e.message}"
            }
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Diagnose") },
                actions = {
                    IconButton(onClick = onHistoryClick) {
                        Icon(Icons.Default.History, "History")
                    }
                }
            )
        }
    ) { padding ->
        Box(modifier = Modifier.fillMaxSize().padding(padding)) {
            if (!cameraPermission.status.isGranted) {
                Column(modifier = Modifier.align(Alignment.Center),
                       horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("Camera permission required")
                    Spacer(Modifier.height(16.dp))
                    Button(onClick = { cameraPermission.launchPermissionRequest() }) {
                        Text("Grant Permission")
                    }
                }
            } else {
                AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())

                Card(
                    modifier = Modifier.align(Alignment.TopCenter).padding(16.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = Color.Black.copy(alpha = 0.7f))
                ) {
                    Column(Modifier.padding(12.dp)) {
                        Text("Crop (optional)", color = Color.White,
                             style = MaterialTheme.typography.labelMedium)
                        Spacer(Modifier.height(6.dp))
                        Row {
                            FilterChip(
                                selected = cropOverride == null,
                                onClick = { cropOverride = null },
                                label = { Text("Auto") },
                                modifier = Modifier.padding(end = 6.dp)
                            )
                            listOf("maize", "tomato", "potato").forEach { c ->
                                FilterChip(
                                    selected = cropOverride == c,
                                    onClick = { cropOverride = c },
                                    label = { Text(c.replaceFirstChar { it.uppercase() }) },
                                    modifier = Modifier.padding(end = 6.dp)
                                )
                            }
                        }
                        Text(
                            if (cropOverride == null) "Mode: Auto-detect"
                            else "Mode: Forced ${cropOverride!!.replaceFirstChar { it.uppercase() }}",
                            color = Color.Yellow,
                            style = MaterialTheme.typography.labelSmall
                        )
                    }
                }

                Button(
                    onClick = {
                        if (isProcessing) return@Button
                        isProcessing = true; errorMsg = null
                        imageCapture.takePicture(
                            ContextCompat.getMainExecutor(context),
                            object : ImageCapture.OnImageCapturedCallback() {
                                override fun onCaptureSuccess(image: ImageProxy) {
                                    try {
                                        val bmp = imageProxyToBitmap(image)

                                        val (isLeaf, reason) = classifier.isLikelyLeaf(bmp)
                                        if (!isLeaf) {
                                            errorMsg = "Not a leaf — $reason. " +
                                                       "Point at a leaf and tap SCAN."
                                            isProcessing = false
                                            image.close()
                                            return
                                        }

                                        val pred = classifier.predict(bmp, cropOverride)

                                        if (pred.rawMaxConfidence < MIN_RAW_CONFIDENCE) {
                                            errorMsg = "Model is uncertain " +
                                                       "(${(pred.rawMaxConfidence * 100).toInt()}%). " +
                                                       "Try better lighting or closer shot."
                                            isProcessing = false
                                            image.close()
                                            return
                                        }

                                        val usedCrop = pred.cropType
                                        scope.launch {
                                            try {
                                                repository.save(bmp, pred, usedCrop)
                                                SyncWorker.syncNow(context)
                                            } catch (e: Exception) {
                                                android.util.Log.e("CameraScreen",
                                                    "Room save failed: ${e.message}")
                                            }
                                            withContext(Dispatchers.Main) {
                                                onPredictionReady(pred)
                                            }
                                        }

                                    } catch (e: Exception) {
                                        errorMsg = "Inference failed: ${e.message}"
                                    } finally {
                                        image.close()
                                        isProcessing = false
                                    }
                                }
                                override fun onError(exc: ImageCaptureException) {
                                    errorMsg = "Capture failed: ${exc.message}"
                                    isProcessing = false
                                }
                            }
                        )
                    },
                    enabled = !isProcessing,
                    modifier = Modifier.align(Alignment.BottomCenter)
                        .padding(24.dp).size(80.dp),
                    shape = RoundedCornerShape(50)
                ) {
                    if (isProcessing) {
                        CircularProgressIndicator(
                            color = MaterialTheme.colorScheme.onPrimary,
                            modifier = Modifier.size(32.dp))
                    } else {
                        Text("SCAN", style = MaterialTheme.typography.labelLarge)
                    }
                }

                errorMsg?.let { msg ->
                    Card(
                        modifier = Modifier.align(Alignment.BottomCenter)
                            .padding(bottom = 130.dp, start = 16.dp, end = 16.dp),
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.errorContainer)
                    ) {
                        Text(msg, modifier = Modifier.padding(12.dp),
                             color = MaterialTheme.colorScheme.onErrorContainer)
                    }
                }
            }
        }
    }
}

private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
    val buffer = image.planes[0].buffer
    val bytes = ByteArray(buffer.remaining()); buffer.get(bytes)
    val bmp = android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    val matrix = Matrix().apply { postRotate(image.imageInfo.rotationDegrees.toFloat()) }
    return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)
}