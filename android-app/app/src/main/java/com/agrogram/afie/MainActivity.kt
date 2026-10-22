package com.agrogram.afie

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.agrogram.afie.ml.DiseaseClassifier
import com.agrogram.afie.ui.CameraScreen
import com.agrogram.afie.ui.HistoryScreen
import com.agrogram.afie.ui.ResultScreen
import com.agrogram.afie.ui.theme.AgroGramAFIETheme
import com.agrogram.afie.worker.SyncWorker

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        SyncWorker.schedule(applicationContext)
        setContent {
            AgroGramAFIETheme {
                AppNavigation()
            }
        }
    }
}

@Composable
fun AppNavigation() {
    val navController = rememberNavController()

    NavHost(navController = navController, startDestination = "camera") {
        composable("camera") {
            CameraScreen(
                onPredictionReady = { result ->
                    navController.currentBackStackEntry?.savedStateHandle?.set("prediction", result)
                    navController.navigate("result")
                },
                onHistoryClick = { navController.navigate("history") }
            )
        }
        composable("result") {
            val prediction = navController.previousBackStackEntry
                ?.savedStateHandle
                ?.get<DiseaseClassifier.Prediction>("prediction")
            if (prediction != null) {
                ResultScreen(
                    prediction = prediction,
                    onBack = { navController.popBackStack() }
                )
            } else {
                Surface(modifier = Modifier.fillMaxSize()) {
                    Box(contentAlignment = Alignment.Center) {
                        Text("No prediction available")
                    }
                }
            }
        }
        composable("history") {
            HistoryScreen(onBack = { navController.popBackStack() })
        }
    }
}