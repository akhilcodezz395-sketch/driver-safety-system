package com.example.driverwarning

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Wrapper for TensorFlow Lite model inference.
 * 
 * Loads curve_detector.tflite from assets and provides prediction API.
 */
class TFLiteModelRunner(context: Context) {
    
    private val interpreter: Interpreter
    
    init {
        val model = loadModelFile(context, "curve_detector.tflite")
        interpreter = Interpreter(model)
    }
    
    /**
     * Load TFLite model from assets.
     */
    private fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    /**
     * Run inference on feature vector.
     * 
     * @param features FloatArray of size 9 (must be normalized)
     * @return FloatArray of size 4 (softmax probabilities: [safe, mild, urgent, hectic])
     */
    fun predict(features: FloatArray): FloatArray {
        require(features.size == 9) { 
            "Expected 9 features, got ${features.size}" 
        }
        
        // Prepare input: shape [1, 9]
        val inputArray = Array(1) { features }
        
        // Prepare output: shape [1, 4]
        val outputArray = Array(1) { FloatArray(4) }
        
        // Run inference
        interpreter.run(inputArray, outputArray)
        
        return outputArray[0]
    }
    
    /**
     * Close the interpreter and free resources.
     */
    fun close() {
        interpreter.close()
    }
}
