using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Mediapipe;
using Mediapipe.Unity;
using Mediapipe.Unity.CoordinateSystem;

using Stopwatch = System.Diagnostics.Stopwatch;

public class HandTracking : MonoBehaviour
{
    [SerializeField] private TextAsset _configAsset;
    [SerializeField] private RawImage _screen;
    [SerializeField] private int _width;
    [SerializeField] private int _height;
    [SerializeField] private int _fps;
    [SerializeField] private GameObject _handPoint;

    private CalculatorGraph _graph;
    private ResourceManager _resourceManager;

    private WebCamTexture _webCamTexture;
    private Texture2D _inputTexture;
    private Color32[] _pixelData;
    private GameObject[] hand;

    
    private IEnumerator Start()
    {
        if (WebCamTexture.devices.Length == 0)
        {
            throw new System.Exception("Web Camera devices are not found");
        }
        // Turn on Webcam
        var webcamDevice = WebCamTexture.devices[0];
        _webCamTexture = new WebCamTexture(webcamDevice.name, _width, _height, _fps);
        _webCamTexture.Play();
        hand = new GameObject[21];
        for (var i = 0; i < 21; i++)
        {
            hand[i] = Instantiate(_handPoint, _screen.transform);
        }

        // return coroutine function
        yield return new WaitUntil(() => _webCamTexture.width > 16);
        //_screen.rectTransform.sizeDelta = new Vector2(_width, _height);
        /*yield return GpuManager.Initialize();

        if (!GpuManager.IsInitialized)
        {
            throw new System.Exception("GPU Resources Could Not Started");
        }*/

        //_screen.rectTransform.sizeDelta = new Vector2(_width, _height);
        //_screen.texture = _webCamTexture;

        // Bring Hands moedel
        _resourceManager = new StreamingAssetsResourceManager();
        yield return _resourceManager.PrepareAssetAsync("hand_landmark_full.bytes");
        yield return _resourceManager.PrepareAssetAsync("hand_landmark_lite.bytes");
        yield return _resourceManager.PrepareAssetAsync("palm_detection_full.bytes");
        yield return _resourceManager.PrepareAssetAsync("palm_detection_lite.bytes");
        yield return _resourceManager.PrepareAssetAsync("hand_recrop.bytes");
        yield return _resourceManager.PrepareAssetAsync("handedness.txt");

        var stopwatch = new Stopwatch();

        // image frame for input stream
        _inputTexture = new Texture2D(_width, _height, TextureFormat.RGBA32, false);
        _pixelData = new Color32[_width * _height];

        // CalculatorGraph initialize
        _graph = new CalculatorGraph(_configAsset.text);
        //_graph.SetGpuResources(GpuManager.GpuResources).AssertOk();

        // Bring landmarks ourputstream
        var handLandmarksStream = new OutputStream<NormalizedLandmarkListVectorPacket, List<NormalizedLandmarkList>>(
            _graph, "hand_landmarks");

        // polling
        handLandmarksStream.StartPolling().AssertOk();

        var sidePacket = new SidePacket();
        sidePacket.Emplace("input_horizontally_flipped", new BoolPacket(true));
        sidePacket.Emplace("input_rotation", new IntPacket(0));
        sidePacket.Emplace("input_vertically_flipped", new BoolPacket(true));
        sidePacket.Emplace("num_hands", new IntPacket(2));
        // Run graph
        _graph.StartRun(sidePacket).AssertOk();

        stopwatch.Start();

        // Bring Rectransform of Screen
        var screenRect = _screen.GetComponent<RectTransform>().rect;

        while (true)
        {
            _inputTexture.SetPixels32(_webCamTexture.GetPixels32(_pixelData));

            // ImageFrame initialize
            var imageFrame = new ImageFrame(
                ImageFormat.Types.Format.Srgba,
                _width, _height,
                _width * 4,
                _inputTexture.GetRawTextureData<byte>());
            // Make Timestamp
            var currentTimestamp = stopwatch.ElapsedTicks / (System.TimeSpan.TicksPerMillisecond / 1000);

            // imageFrame을 packet에 싸서 inputstream으로 보내기
            _graph.AddPacketToInputStream("input_video", new ImageFramePacket(imageFrame, new Timestamp(currentTimestamp))).AssertOk();

            yield return new WaitForEndOfFrame();

            // Get landmarks values
            if (handLandmarksStream.TryGetNext(out var handLandmarks))
            {
                foreach (var landmarks in handLandmarks)
                {
                    for (var i = 0; i < landmarks.Landmark.Count; i++)
                    {
                        var worldLandmarkPos = screenRect.GetPoint(landmarks.Landmark[i]);
                        hand[i].transform.localPosition = worldLandmarkPos;
                    }
                }
            }
        }
    }

    private void OnDestroy()
    {
        if (_webCamTexture != null)
        {
            _webCamTexture.Stop();
        }
        if (_graph != null)
        {
            try
            {
                _graph.CloseInputStream("input_video").AssertOk();
                _graph.WaitUntilDone().AssertOk();
            }
            finally
            {
                _graph.Dispose();
            }
        }
        GpuManager.Shutdown();
    }
}