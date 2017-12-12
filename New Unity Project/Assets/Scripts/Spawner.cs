using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Spawner : MonoBehaviour {

	public Transform SpawnerTransform;
	public GameObject Spawnee;
	public GameObject drone;
	public Transform Player;
	public Vector3 playerPos;
	public float coordsX;

	private Rigidbody SpawneeRb;
	private float initSpeedX;
	private float initSpeedY;
	private float initSpeedZ;

	// Use this for initialization
	void Start () {
		SpawnerTransform = gameObject.transform;
		SpawneeRb = Spawnee.GetComponent<Rigidbody> ();
		InvokeRepeating ("SpawnObstacle", 3, 3);
	}


	void SpawnObstacle() {
		Rigidbody clone;
		Player = drone.transform;
		playerPos = Player.position;
		coordsX = SpawnerTransform.position.x + Random.Range (-10.0f, 10.0f);
		clone = Instantiate (SpawneeRb, new Vector3(coordsX, SpawnerTransform.position.y, SpawnerTransform.position.z) , Quaternion.identity) as Rigidbody;

		initSpeedZ = -25.0f;
		initSpeedX = (playerPos.x + Random.Range(-1.0f, 1.0f) - coordsX) * initSpeedZ / (Player.position.z - SpawnerTransform.position.z);
		initSpeedY = -Physics.gravity.y * (Player.position.z - SpawnerTransform.position.z) / (2 * initSpeedZ);
		clone.velocity = new Vector3(initSpeedX, initSpeedY, initSpeedZ);
		// obstacleVelocity = clone.velocity;


	}
}
