using UnityEngine;
using System.Collections;
using UnityEngine.SceneManagement;
using System.Collections.Generic;
using SocketIO;

public class cubemovement : MonoBehaviour {

	public Rigidbody rb;
	public int action { get; set; }
	//public int action;
	public int movement;
	public float speed;
	public int count = 0;

	// Use this for initialization
	void Start () {
		rb = gameObject.GetComponent<Rigidbody> ();		
	}

	// Update is called once per frame
	void Update () {
		if (Input.GetKeyDown ("a")) {
			print ("hi");
			SocketIOComponent.Instance.Connect();
		}
	}

	void FixedUpdate () {
		if (count == 0) {
			action = Random.Range (0, 2);
			count = 4;
		} else {
			if (action == 0) {
				movement = -1;
				count -= 1;
			}
			if (action == 1) {
				movement = 1;
				count -= 1;
			}
		} 

		rb.velocity = new Vector3 (movement * speed, 0, 0);
	}

	void OnTriggerEnter(Collider other) {
		if (other.gameObject.CompareTag ("Obstacle")) {
			Destroy (other.gameObject);
			rb.position = new Vector3 (0.0f, 3.0f, -4.0f);
		}
	}
}
