using UnityEngine;
using System.Collections;
using UnityEngine.SceneManagement;
using System.Collections.Generic;
using SocketIO;

public class cubemovement : MonoBehaviour {

	public Rigidbody rb;
	public int action { get; set; }
	public int Left_Changing = 0;
	public int Right_Changing = 0;
	//public int action; //when player plays
	public int movement;
	public float speed;
	public int count = 0;


	private IEnumerator Start()
	{
		//DontDestroyOnLoad(this.gameObject);
		yield return new WaitForSeconds(0.5f);
		CommandServer.Instance.Init();
		CommandServer.Instance.EmitTelemetry();
		rb = gameObject.GetComponent<Rigidbody>();
	}

	void FixedUpdate () {
		if (count == 0) {
			//action = Random.Range (0, 2);  // when player plays
			count = 4;
		} else {
			if (action == 0) {
				movement = 0;
				count -= 1;
			}
			if (action == 1) {
				movement = -1;
				Left_Changing = 1;
				count -= 1;
			}
			if (action == 2) {
				movement = 1;
				Right_Changing = 1;
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
	private void Update()
	{
		if (action == 1)
		{
			print("Connected!!");
		}
	}
}
