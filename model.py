from ultralytics import YOLO
import cv2
import math
import pandas as pd
import numpy as np
import os
from collections import defaultdict


def process_video(input_path, output_path, excel_path, frames_dir=None):
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return []

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0:
        fps = 25
    if total_frames <= 0:
        total_frames = 300

    # Use 'avc1' for H.264 (most compatible with browsers)
    # If it fails, fallback to 'mp4v'
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # ─── Lane Configuration ───
    # Divide road into 3 virtual lanes
    lane_count = 3
    lane_width = w // lane_count
    lane_boundaries = [lane_width * i for i in range(1, lane_count)]  # x positions of lane dividers
    lane_names = ["Left Lane", "Center Lane", "Right Lane"]

    # ─── Speed & Safety Config ───
    meter_per_pixel = 0.05
    speed_limit = 60          # km/h — above this = DANGER
    warning_speed = 45        # km/h — above this = WARNING
    danger_distance = 100     # pixels — vehicles too close = DANGER
    warning_distance = 180    # pixels — vehicles somewhat close = WARNING
    speed_history_len = 5     # frames to average speed over
    MIN_VISIBILITY_FRAMES = 3   # Minimum frames a vehicle must be visible to be counted (Lowered from 10)

    # ─── Tracking State ───
    logs = []
    prev_positions = {}
    speed_histories = defaultdict(list)       # tracker_id → list of recent speeds
    tracker_to_number = {}                     # tracker_id → sequential number (1,2,3...)
    next_vehicle_number = 1
    vehicle_data = defaultdict(lambda: {      # per-vehicle aggregated data
        "type": "Vehicle",
        "speeds": [],
        "lanes": [],
        "statuses": [],
        "first_frame": 0,
        "last_frame": 0,
    })

    frame_no = 0
    saved_frames = 0
    frame_interval = max(1, total_frames // 6)

    # ─── Colors ───
    COLOR_SAFE = (0, 200, 0)         # Green
    COLOR_WARNING = (0, 220, 255)    # Yellow
    COLOR_DANGER = (0, 0, 255)       # Red

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1

        # ─── Draw Virtual Lane Lines ───
        overlay = frame.copy()
        for bx in lane_boundaries:
            cv2.line(overlay, (bx, 0), (bx, h), (255, 200, 0), 2, cv2.LINE_AA)
        # Lane labels at top
        for li, ln in enumerate(lane_names):
            lx = li * lane_width + lane_width // 2
            text_size = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
            cv2.putText(overlay, ln, (lx - text_size[0] // 2, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1, cv2.LINE_AA)
        # Blend overlay so lane lines are semi-transparent
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # ─── YOLO Detection + Tracking ───
        results = model.track(frame, persist=True, verbose=False)

        # ─── 1. Gather all detections in this frame first ───
        frame_detections = []
        for r in results:
            if r.boxes is None: continue
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            ids = r.boxes.id
            if ids is None: continue
            ids = ids.cpu().numpy()

            for i, box in enumerate(boxes):
                cls = int(classes[i])
                if cls not in [2, 3, 5, 7]: continue
                
                vehicle_type_map = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
                v_type = vehicle_type_map.get(cls, "Vehicle")
                tracker_id = int(ids[i])
                
                if tracker_id not in tracker_to_number:
                    tracker_to_number[tracker_id] = next_vehicle_number
                    next_vehicle_number += 1
                v_num = tracker_to_number[tracker_id]

                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Speed Calculation
                raw_speed = 0.0
                if tracker_id in prev_positions:
                    px, py = prev_positions[tracker_id]
                    pixel_dist = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                    raw_speed = pixel_dist * meter_per_pixel * fps * 3.6
                prev_positions[tracker_id] = (cx, cy)

                speed_histories[tracker_id].append(raw_speed)
                if len(speed_histories[tracker_id]) > speed_history_len:
                    speed_histories[tracker_id] = speed_histories[tracker_id][-speed_history_len:]
                speed = sum(speed_histories[tracker_id]) / len(speed_histories[tracker_id])

                frame_detections.append({
                    "id": tracker_id, "num": v_num, "type": v_type,
                    "box": (x1, y1, x2, y2), "center": (cx, cy),
                    "speed": speed, "lane": min(cx // lane_width, lane_count - 1),
                    "proximity_status": "SAFE"
                })

        # ─── 2. Proximity Check ───
        for i in range(len(frame_detections)):
            for j in range(i + 1, len(frame_detections)):
                d1, d2 = frame_detections[i], frame_detections[j]
                dist = math.sqrt((d1["center"][0] - d2["center"][0])**2 + (d1["center"][1] - d2["center"][1])**2)
                
                if dist < danger_distance:
                    d1["proximity_status"] = "DANGER"
                    d2["proximity_status"] = "DANGER"
                    cv2.line(frame, d1["center"], d2["center"], COLOR_DANGER, 3)
                elif dist < warning_distance:
                    if d1["proximity_status"] != "DANGER": d1["proximity_status"] = "WARNING"
                    if d2["proximity_status"] != "DANGER": d2["proximity_status"] = "WARNING"
                    cv2.line(frame, d1["center"], d2["center"], COLOR_WARNING, 2)

        # ─── 3. Final Status & Drawing ───
        unique_ids_in_frame = set()
        for d in frame_detections:
            tid = d["id"]
            unique_ids_in_frame.add(tid)
            
            # Speed status
            status = "SAFE"
            if d["speed"] > speed_limit: status = "DANGER"
            elif d["speed"] > warning_speed: status = "WARNING"
            
            # Combine with proximity (worst case wins)
            if d["proximity_status"] == "DANGER" or status == "DANGER":
                status = "DANGER"
                color = COLOR_DANGER
                risk = "High"
            elif d["proximity_status"] == "WARNING" or status == "WARNING":
                status = "WARNING"
                color = COLOR_WARNING
                risk = "Medium"
            else:
                status = "SAFE"
                color = COLOR_SAFE
                risk = "Low"

            x1, y1, x2, y2 = d["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, d["center"], 4, color, -1)

            lane_name = lane_names[d["lane"]]
            label = f"#{d['num']} {d['type']} | {d['speed']:.0f}km/h"
            label2 = f"{lane_name} | {status}"
            (tw1, th1), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            (tw2, th2), _ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - th1 - th2 - 16), (x1 + max(tw1, tw2) + 8, y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - th2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, label2, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # Update aggregation
            vd = vehicle_data[tid]
            vd["type"] = d["type"]
            vd["speeds"].append(d["speed"])
            vd["lanes"].append(lane_name)
            vd["statuses"].append(status)
            if vd["first_frame"] == 0: vd["first_frame"] = frame_no
            vd["last_frame"] = frame_no

            logs.append({
                "Frame": frame_no, "Vehicle_#": d["num"], "Vehicle_ID": tid,
                "Vehicle_Type": d["type"], "Speed_kmph": round(d["speed"], 2),
                "Lane": lane_name, "Status": status, "Accident_Risk": risk
            })

        # ─── Danger Banner ───
        danger_pairs = sum(
            1 for i in range(len(frame_detections))
            for j in range(i + 1, len(frame_detections))
            if math.sqrt((frame_detections[i]["center"][0] - frame_detections[j]["center"][0]) ** 2 + 
                         (frame_detections[i]["center"][1] - frame_detections[j]["center"][1]) ** 2) < danger_distance
        )
        if danger_pairs > 0:
            cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 180), -1)
            cv2.putText(frame, f"  DANGER: {danger_pairs} VEHICLE(S) TOO CLOSE!",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        # ─── Bottom HUD ───
        hud_h = 40
        cv2.rectangle(frame, (0, h - hud_h), (w, h), (0, 0, 0), -1)
        total_detected = next_vehicle_number - 1
        hud_text = f"Total Vehicles: {total_detected}  |  Active: {len(unique_ids_in_frame)}  |  Frame: {frame_no}/{total_frames}"
        cv2.putText(frame, hud_text, (15, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        out.write(frame)

        # Save snapshot frames
        if frames_dir and (frame_no == 1 or frame_no % frame_interval == 0) and saved_frames < 6:
            frame_path = os.path.join(frames_dir, f"frame_{saved_frames + 1}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_frames += 1

    cap.release()
    out.release()

    # ─── Build Vehicle Summary ───
    vehicle_summary = []
    for tracker_id, vd in vehicle_data.items():
        v_num = tracker_to_number.get(tracker_id, 0)
        speeds = vd["speeds"]
        avg_speed = round(sum(speeds) / len(speeds), 1) if speeds else 0
        max_speed = round(max(speeds), 1) if speeds else 0

        # Most frequent lane
        from collections import Counter
        lane_counts = Counter(vd["lanes"])
        primary_lane = lane_counts.most_common(1)[0][0] if lane_counts else "Unknown"

        # Overall status logic:
        # Instead of "worst case wins" immediately, let's use a persistence threshold
        # to avoid marking a vehicle as DANGER just for a single noisy frame.
        status_counts = Counter(vd["statuses"])
        total_status_entries = len(vd["statuses"])
        
        danger_ratio = status_counts["DANGER"] / total_status_entries if total_status_entries > 0 else 0
        warning_ratio = status_counts["WARNING"] / total_status_entries if total_status_entries > 0 else 0

        # Thresholds: If in danger for > 10% of time or > 5 frames
        if status_counts["DANGER"] >= 5 or danger_ratio > 0.1:
            overall_status = "DANGER"
            accident_risk = "High"
        elif status_counts["WARNING"] >= 5 or warning_ratio > 0.1:
            overall_status = "WARNING"
            accident_risk = "Medium"
        else:
            overall_status = "SAFE"
            accident_risk = "Low"

        # Detect lane weaving (if switches lanes more than 3 times)
        unique_lanes = len(set(vd["lanes"]))
        if unique_lanes >= 2 and overall_status == "SAFE":
            overall_status = "WARNING"
            accident_risk = "Medium"

        vehicle_summary.append({
            "Vehicle_#": v_num,
            "Vehicle_Type": vd["type"],
            "Avg_Speed_kmph": avg_speed,
            "Max_Speed_kmph": max_speed,
            "Lane": primary_lane,
            "Lanes_Used": unique_lanes,
            "Status": overall_status,
            "Accident_Risk": accident_risk,
            "Frames_Visible": vd["last_frame"] - vd["first_frame"] + 1
        })

    # Sort by vehicle number
    vehicle_summary.sort(key=lambda x: x["Vehicle_#"])

    # ─── Filter Noise and Re-index ───
    # Keep only vehicles visible for >= MIN_VISIBILITY_FRAMES
    valid_summary = [v for v in vehicle_summary if v["Frames_Visible"] >= MIN_VISIBILITY_FRAMES]
    
    # Map old tracker_id/number to new sequential number
    old_to_new_vnum = {}
    for i, v in enumerate(valid_summary):
        old_vnum = v["Vehicle_#"]
        new_vnum = i + 1
        old_to_new_vnum[old_vnum] = new_vnum
        v["Vehicle_#"] = new_vnum  # Update summary

    vehicle_summary = valid_summary
    valid_v_nums = set(old_to_new_vnum.keys())
    
    # Filter and update logs
    filtered_logs = []
    for log in logs:
        v_num_raw = log["Vehicle_#"]
        
        # Handle "Vehicle Pair" logs (e.g., "#1-#2")
        if log["Vehicle_Type"] == "Vehicle Pair":
            try:
                # Extract numbers from "#1-#2"
                parts = v_num_raw.replace("#", "").split("-")
                num1, num2 = int(parts[0]), int(parts[1])
                if num1 in valid_v_nums and num2 in valid_v_nums:
                    log["Vehicle_#"] = f"#{old_to_new_vnum[num1]}-#{old_to_new_vnum[num2]}"
                    # Also update ID if present
                    id_parts = str(log["Vehicle_ID"]).split("-")
                    log["Vehicle_ID"] = f"{id_parts[0]}-{id_parts[1]}"
                    filtered_logs.append(log)
            except:
                continue
        else:
            # Normal vehicle logs
            if v_num_raw in valid_v_nums:
                log["Vehicle_#"] = old_to_new_vnum[v_num_raw]
                filtered_logs.append(log)
    
    logs = filtered_logs

    # ─── Save Excel with 2 Sheets ───
    df_logs = pd.DataFrame(logs)
    df_summary = pd.DataFrame(vehicle_summary)

    if df_logs.empty:
        df_logs = pd.DataFrame([{
            "Frame": 0, "Vehicle_#": 0, "Vehicle_ID": 0,
            "Vehicle_Type": "None", "Speed_kmph": 0,
            "Lane": "-", "Status": "No vehicle detected",
            "Accident_Risk": "-"
        }])

    if df_summary.empty:
        df_summary = pd.DataFrame([{
            "Vehicle_#": 0, "Vehicle_Type": "None",
            "Avg_Speed_kmph": 0, "Max_Speed_kmph": 0,
            "Lane": "-", "Lanes_Used": 0,
            "Status": "No vehicle detected",
            "Accident_Risk": "-", "Frames_Visible": 0
        }])

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_logs.to_excel(writer, sheet_name='Frame Logs', index=False)
        df_summary.to_excel(writer, sheet_name='Vehicle Summary', index=False)

    print("Video saved:", output_path)
    print("Excel saved:", excel_path)
    print(f"Total vehicles detected: {len(vehicle_summary)}")

    return vehicle_summary
