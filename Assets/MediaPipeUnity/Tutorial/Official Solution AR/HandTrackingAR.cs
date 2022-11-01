/*
* Rastramento Básico de Mão com MediaPipe Hands com Retorno das Coordenadas com uso de ARFoundation *

- Utilizando uma ARCamera e retornando as coordenadas "hand_landmarks" (normalizadas) no console...
- Utilização de GPU para processamento, com configurações padrões da documentação oficial (TFFull).
- Processamento no modo ASSÍNCRONO para MediaPipe (obrigatório com ARFoundation): https://github.com/homuler/MediaPipeUnityPlugin/wiki/Advanced-Topics
- Processamento no modo SÍNCRONO para ARFoundation: https://docs.unity3d.com/Packages/com.unity.xr.arfoundation@4.2/manual/cpu-camera-image.html

-- Info sobre o "hand_world_landmarks": https://github.com/google/mediapipe/issues/2199#issuecomment-1002299634

By Davi Neves => davimedio01 // 2022
*/

using System.Collections;
using System.Collections.Generic;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

using Mediapipe;
using Mediapipe.Unity;
using Mediapipe.Unity.CoordinateSystem;

using Stopwatch = System.Diagnostics.Stopwatch;
using System;

public class HandTrackingAR : MonoBehaviour
{
    private enum InferenceMode
    {
        CPU,
        GPU
    }
    private enum ModelComplexity
    {
        Lite = 0,
        Full = 1,
    }

    private enum MaxNumberHands
    {
        One = 1,
        Two = 2,
    }

    [SerializeField] private TextAsset _cpuConfig;
    [SerializeField] private TextAsset _gpuConfig;
    [SerializeField] private TextAsset _openGlEsConfig;
    [SerializeField] private InferenceMode _inferenceMode = InferenceMode.GPU;
    private TextAsset _configAsset;

    [SerializeField] private ModelComplexity _modelComplexity = ModelComplexity.Full;
    [SerializeField] private MaxNumberHands _maxNumHands = MaxNumberHands.One;

    [SerializeField] private RawImage _screen;
    [SerializeField] private ARCameraManager _cameraManager;
    [SerializeField] private GameObject _handPoint;

    private CalculatorGraph _graph;
    private static UnityEngine.Rect _screenRect;
    private GameObject[] hand;
    private Stopwatch _stopwatch;
    private ResourceManager _resourceManager;
    private GpuResources _gpuResources;
    private NativeArray<byte> _buffer;
    private const string _InputStreamName = "input_video";
    private const string _SidePacketModelComplexity = "model_complexity";
    private const string _SidePacketMaxHandsName = "num_hands";

    private OutputStream<NormalizedLandmarkListVectorPacket, List<NormalizedLandmarkList>> _handLandmarkStream;

    private IEnumerator Start()
    {
        Glog.Logtostderr = true;
        Glog.Minloglevel = 0;
        Glog.V = 3;
        Protobuf.SetLogHandler(Protobuf.DefaultLogHandler);

        _cameraManager.frameReceived += OnCameraFrameReceived;
        _gpuResources = (_inferenceMode == InferenceMode.GPU) ? GpuResources.Create().Value() : null;
        hand = new GameObject[21];
        for (var i = 0; i < 21; i++)
        {
            hand[i] = Instantiate(_handPoint, _screen.transform);
        }

        _resourceManager = new StreamingAssetsResourceManager();
        if (_modelComplexity == ModelComplexity.Lite)
        {
            yield return _resourceManager.PrepareAssetAsync("hand_landmark_lite.bytes");
            yield return _resourceManager.PrepareAssetAsync("hand_recrop.bytes");
            yield return _resourceManager.PrepareAssetAsync("handedness.txt");
            yield return _resourceManager.PrepareAssetAsync("palm_detection_lite.bytes");
        }
        else
        {
            yield return _resourceManager.PrepareAssetAsync("hand_landmark_full.bytes");
            yield return _resourceManager.PrepareAssetAsync("hand_recrop.bytes");
            yield return _resourceManager.PrepareAssetAsync("handedness.txt");
            yield return _resourceManager.PrepareAssetAsync("palm_detection_full.bytes");
        }

        _stopwatch = new Stopwatch();

        if (SystemInfo.deviceType == DeviceType.Handheld)
        {
            _configAsset = _gpuConfig;
            Debug.Log("DeviceType: " + SystemInfo.deviceType);
        }
        else if (SystemInfo.deviceType == DeviceType.Desktop)
        {
            if (_inferenceMode == InferenceMode.CPU)
            {
                _configAsset = _cpuConfig;
            }
            else if (_inferenceMode == InferenceMode.GPU)
            {
                _configAsset = _gpuConfig;
            }
        }
        else
        {
            _configAsset = _gpuConfig;
        }

        _graph = new CalculatorGraph(_configAsset.text);
        if (_gpuResources != null)
            _graph.SetGpuResources(_gpuResources).AssertOk();
        _screenRect = _screen.GetComponent<RectTransform>().rect;

        _handLandmarkStream = new OutputStream<NormalizedLandmarkListVectorPacket, List<NormalizedLandmarkList>>(
            _graph, "hand_landmarks");

        _handLandmarkStream.StartPolling().AssertOk();

        var sidePacket = new SidePacket();
        sidePacket.Emplace(_SidePacketModelComplexity, new IntPacket((int)_modelComplexity));
        sidePacket.Emplace(_SidePacketMaxHandsName, new IntPacket((int)_maxNumHands));
        sidePacket.Emplace("input_rotation", new IntPacket(90));
        sidePacket.Emplace("input_horizontally_flipped", new BoolPacket(true));
        sidePacket.Emplace("input_vertically_flipped", new BoolPacket(true));

        _graph.StartRun(sidePacket).AssertOk();
        _stopwatch.Start();
    }

    private unsafe void OnCameraFrameReceived(ARCameraFrameEventArgs eventArgs)
    {
        if (_cameraManager.TryAcquireLatestCpuImage(out XRCpuImage image))
        {
            alocBuffer(image);
            var conversionParams = new XRCpuImage.ConversionParams(image, TextureFormat.RGBA32);
            var ptr = (IntPtr)NativeArrayUnsafeUtility.GetUnsafePtr(_buffer);
            image.Convert(conversionParams, ptr, _buffer.Length);
            image.Dispose();
            var imageFrame = new ImageFrame(ImageFormat.Types.Format.Srgba, image.width, image.height, 4 * image.width, _buffer);
            var currentTimestamp = _stopwatch.ElapsedTicks / (TimeSpan.TicksPerMillisecond / 1000);
            var imageFramePacket = new ImageFramePacket(imageFrame, new Timestamp(currentTimestamp));
            _graph.AddPacketToInputStream(_InputStreamName, imageFramePacket).AssertOk();
            StartCoroutine(WaitForEndOfFrameCoroutine());
            if (_handLandmarkStream.TryGetNext(out var handLandmarks))
            {
                foreach (var landmarks in handLandmarks)
                {
                    for (var i = 0; i < landmarks.Landmark.Count; i++)
                    {
                        var worldLandmarkPos = _screenRect.GetPoint(landmarks.Landmark[i]);
                        hand[i].transform.localPosition = worldLandmarkPos;
                    }
                }
            }
        }
    }

    private IEnumerator WaitForEndOfFrameCoroutine()
    {
        yield return new WaitForEndOfFrame();
    }

    private void alocBuffer(XRCpuImage image)
    {
        var length = image.width * image.height * 4;
        if (_buffer == null || _buffer.Length != length)
        {
            _buffer = new NativeArray<byte>(length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        }
    }

    private void OnDestroy()
    {
        _cameraManager.frameReceived -= OnCameraFrameReceived;
        var statusGraph = _graph.CloseAllPacketSources();
        if (!statusGraph.Ok())
        {
            Debug.Log($"Failed to close packet sources: {statusGraph}");
        }

        statusGraph = _graph.WaitUntilDone();
        if (!statusGraph.Ok())
        {
            Debug.Log(statusGraph);
        }

        _graph.Dispose();

        if (_gpuResources != null)
            _gpuResources.Dispose();

        _buffer.Dispose();
    }

    private void OnApplicationQuit()
    {
        Protobuf.ResetLogHandler();
    }
}