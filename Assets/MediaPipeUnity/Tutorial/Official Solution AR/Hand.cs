using Mediapipe;
using UnityEngine;

public class Hand : MonoBehaviour
{
    [SerializeField] private GameObject[] _handPoints;

    public void SetHandPoint(Vector3 landmarkPos, int point)
    {
        _handPoints[point].transform.position = landmarkPos;
    }
}
