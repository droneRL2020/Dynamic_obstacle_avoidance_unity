using UnityEngine;
using System.Collections.Generic;
using System.Collections;
using SocketIO;
using UnityStandardAssets.Vehicles.Car;
using System;
using System.Security.AccessControl;

public class CommandServer : MonoBehaviour
{

	public static CommandServer _instance;
	public static CommandServer Instance {
		get{
			if (!_instance) {
				GameObject go = new GameObject ();
				go.name = "CommandServer";
				_instance = go.AddComponent (typeof(CommandServer)) as CommandServer;
			}
			return _instance;
		}
	}

	public cubemovement HostMovement;
	public Camera FrontFacingCamera;
//	public Camera RearFacingCamera;
	private cubemovement _hostMovement;

	private String Front_camera_old;
//	private String Rear_camera_old;
//
	private String Front_cam_data;
//	private String Rear_cam_data;

	// Use this for initialization
	void Start()
	{
		SocketIOComponent.Instance.On("open", OnOpen);
		SocketIOComponent.Instance.On("onsteer", steer);
		SocketIOComponent.Instance.On("manual", onManual);

		Init ();

		// If comment this, disconnection happens
		DontDestroyOnLoad (this);
	}

	public void Init(){
		
		HostMovement = GameObject.FindWithTag ("Player").GetComponent<cubemovement> ();
		FrontFacingCamera = GameObject.FindWithTag("Camera").GetComponent<Camera> ();
//		RearFacingCamera = GameObject.Find("BackwardViewCam").GetComponent<Camera> ();
		_hostMovement = HostMovement.GetComponent<cubemovement>();


	}

	// Update is called once per frame
	void Update()
	{
		if (FrontFacingCamera) {
			Front_cam_data = Convert.ToBase64String (CameraHelper.CaptureFrame (FrontFacingCamera));
//			Rear_cam_data = Convert.ToBase64String (CameraHelper.CaptureFrame (RearFacingCamera));
		}
	}

	void OnOpen(SocketIOEvent obj)
	{
		Debug.Log("Connection Open");
		//EmitTelemetry();
	}

	// 
	void onManual(SocketIOEvent obj)
	{
		//EmitTelemetry ();
	}
//
	void steer(SocketIOEvent obj)
	{
		JSONObject jsonObject = obj.data;
		HostMovement.action = int.Parse (jsonObject.GetField ("action").str);

		EmitTelemetry();
	}

	public void EmitTelemetry()
	{
		//UnityMainThreadDispatcher.Instance().Enqueue(() =>
		//{
			//print("Attempting to Send...");
			// send only if it's not being manually driven
			//if ((Input.GetKey(KeyCode.W)) || (Input.GetKey(KeyCode.S))) {
			//	_socket.Emit("telemetry", new JSONObject());
			//}
			//else {
				// Collect Data from the Car
				Dictionary<string, string> data = new Dictionary<string, string>();
				//List<float> data_Lidar = new List<float>();
				
				data ["test"] = 0.123.ToString ("N4");
//				data["Speed"] = _hostMovement.m_Speed.ToString ("N4");
				data["Action_vehicle"] = _hostMovement.action.ToString("N4");
//				//data["reward"] = _hostMovement.reward.ToString("N4");
//				data["Front_ADAS"] = _hostMovement.Front_ADAS.ToString("N4");
//				data["Left_ADAS"] = _hostMovement.Left_ADAS.ToString("N4");
//				data["Right_ADAS"] = _hostMovement.Right_ADAS.ToString("N4");
				data["Left_Changing"] = _hostMovement.Left_Changing.ToString("N4");
				data["Right_Changing"] = _hostMovement.Right_Changing.ToString("N4");
//				data["Vehicle_X"] = _hostMovement.Vehicle_x.ToString("N4");
//				data["Vehicle_Z"] = _hostMovement.Vehicle_z.ToString ("N4");

				//data["terminal"] = _hostMovement.terminal.ToString("N4");

//				if (HostMovement.Range_list.Count < 360) {
//					for (int i = 0; i < 360; i++){
//						data[i.ToString("D")] = 0.ToString("N4");
//					}
//				} else {
//					for (int i = 0; i < 360; i++){
//						data[i.ToString("D")] = HostMovement.Range_list[i].ToString("N4");
//					}
//				}
				if (FrontFacingCamera != null) {

					//data["front_image"] = Convert.ToBase64String(CameraHelper.CaptureFrame(FrontFacingCamera));
					data["front_image"] = Front_cam_data;
					
					//Front_camera_old = Convert.ToBase64String(CameraHelper.CaptureFrame(FrontFacingCamera));
					Front_camera_old = Front_cam_data;
					
				} else {
					data["front_image"] = Front_camera_old;
				}
				


				//data_Lidar = HostMovement.Range_list.ToString("N4");
				SocketIOComponent.Instance.Emit("telemetry", new JSONObject(data));
				//_socket.Emit("telemetry", new JSONObject.Type.));
			//}
		//});

	}

	public void CloseSocket(){
		SocketIOComponent.Instance.Close ();
	}
}