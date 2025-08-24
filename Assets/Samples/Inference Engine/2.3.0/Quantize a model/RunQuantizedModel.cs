using UnityEngine;
using Unity.InferenceEngine;

public class RunQuantizedModel : MonoBehaviour
{
    // 1. Download a .sentis or .onnx model you want to quantize and bring into your Unity Project.
    // 2. Open the Editor window 'Inference Engine > Sample > Quantize and Save Model' and reference your model as the source model.
    // 3. Select the desired quantization type and click 'Quantize and Save'.

    // Reference your quantized tiny stories here in the RunQuantizedModel scene.
    [SerializeField]
    ModelAsset modelAsset;
    Worker m_Worker;
    Tensor m_Input;

    const int maxTokens = 100;

    void OnEnable()
    {
        // Load the quantized model as any other Inference Engine model.
        var model = ModelLoader.Load(modelAsset);
        m_Worker = new Worker(model, BackendType.GPUCompute);

        // Initialize input.
        m_Input = new Tensor<int>(new TensorShape(1, maxTokens));
    }

    void Update()
    {
        // Execute worker and peek output as with any other Inference Engine model.
        m_Worker.Schedule(m_Input);
        var output = m_Worker.PeekOutput() as Tensor<float>;
    }

    void OnDisable()
    {
        // Clean up Inference Engine resources.
        m_Worker.Dispose();
        m_Input.Dispose();
    }
}
