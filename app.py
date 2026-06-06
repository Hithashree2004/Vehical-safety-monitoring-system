from flask import Flask, render_template, request, send_file
import os
import pandas as pd
import json
import shutil
from model import process_video
from database import init_db, add_history, update_history, get_history, get_history_by_id

app = Flask(__name__)
init_db()

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/")
def index():
    history = get_history()
    return render_template("index.html", history=history)


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["video"]

    # Create unique folder per analysis
    history_id = add_history(file.filename)
    analysis_dir = os.path.join(OUTPUT_FOLDER, str(history_id))
    frames_dir = os.path.join(analysis_dir, "frames")

    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    output_path = os.path.join(analysis_dir, "output.mp4")
    excel_path = os.path.join(analysis_dir, "traffic_logs.xlsx")

    file.save(input_path)

    # process_video now returns vehicle_summary
    vehicle_summary = process_video(input_path, output_path, excel_path, frames_dir)

    # Also copy to default location for backward compat
    default_output = os.path.join(OUTPUT_FOLDER, "output.mp4")
    default_excel = os.path.join(OUTPUT_FOLDER, "traffic_logs.xlsx")
    shutil.copy2(output_path, default_output)
    shutil.copy2(excel_path, default_excel)

    # Extract real data
    chart_data = extract_chart_data(excel_path)

    # Attach vehicle summary to chart_data for storage
    chart_data["vehicle_summary"] = vehicle_summary if vehicle_summary else []

    # Sync total_vehicles with vehicle_summary if available
    if vehicle_summary:
        chart_data["total_vehicles"] = len(vehicle_summary)

    # Update database with results
    update_history(history_id, chart_data["total_vehicles"], chart_data)

    frames = sorted(os.listdir(frames_dir))

    return render_template("result.html,
                           analysis_id=history_id,
                           frames=frames,
                           total_vehicles=chart_data["total_vehicles"],
                           vehicle_summary=json.dumps(vehicle_summary if vehicle_summary else []),
                           chart_count_data=json.dumps(chart_data["count"]),
                           chart_speed_data=json.dumps(chart_data["speed"]),
                           chart_type_data=json.dumps(chart_data["types"]),
                           chart_danger_data=json.dumps(chart_data["danger"]),
                           chart_lane_data=json.dumps(chart_data["lanes"]),
                           chart_risk_data=json.dumps(chart_data["risk"]))


@app.route("/result/<int:history_id>")
def view_result(history_id):
    entry = get_history_by_id(history_id)
    if not entry:
        return "Analysis not found", 404

    analysis_dir = os.path.join(OUTPUT_FOLDER, str(history_id))
    frames_dir = os.path.join(analysis_dir, "frames")

    if not os.path.exists(analysis_dir):
        return "Analysis files not found", 404

    frames = sorted(os.listdir(frames_dir)) if os.path.exists(frames_dir) else []

    cd = entry.get("chart_data", {})

    return render_template("result.html",
                           analysis_id=history_id,
                           frames=frames,
                           total_vehicles=entry["total_vehicles"],
                           vehicle_summary=json.dumps(cd.get("vehicle_summary", [])),
                           chart_count_data=json.dumps(cd.get("count", {"labels": [], "data": []})),
                           chart_speed_data=json.dumps(cd.get("speed", {"labels": [], "data": []})),
                           chart_type_data=json.dumps(cd.get("types", {"labels": [], "data": []})),
                           chart_danger_data=json.dumps(cd.get("danger", {"labels": ["Too Close", "Overspeed"], "data": [0, 0]})),
                           chart_lane_data=json.dumps(cd.get("lanes", {"labels": [], "data": []})),
                           chart_risk_data=json.dumps(cd.get("risk", {"labels": ["Low", "Medium", "High"], "data": [0, 0, 0]})))


def extract_chart_data(excel_path):
    """Extract all chart data from the Excel file."""
    result = {
        "total_vehicles": 0,
        "count": {"labels": [], "data": []},
        "speed": {"labels": [], "data": []},
        "types": {"labels": [], "data": []},
        "danger": {"labels": ["Too Close", "Overspeed"], "data": [0, 0]},
        "lanes": {"labels": [], "data": []},
        "risk": {"labels": ["Low", "Medium", "High"], "data": [0, 0, 0]},
        "vehicle_summary": []
    }

    try:
        df = pd.read_excel(excel_path, sheet_name='Frame Logs')

        if 'Frame' in df.columns and 'Vehicle_ID' in df.columns:
            counts = df.groupby('Frame')['Vehicle_ID'].nunique().to_dict()
            labels = list(counts.keys())
            data = list(counts.values())
            if len(labels) > 50:
                step = len(labels) // 50
                labels = labels[::step]
                data = data[::step]
            result["count"] = {"labels": [int(l) for l in labels], "data": data}
            # Use unique vehicle IDs from Frame Logs as a baseline
            vehicle_rows = df[df['Vehicle_Type'] != 'Vehicle Pair']
            result["total_vehicles"] = int(vehicle_rows['Vehicle_ID'].nunique()) if not vehicle_rows.empty else 0

        if 'Speed_kmph' in df.columns and 'Vehicle_Type' in df.columns:
            speed_df = df[df['Vehicle_Type'] != 'Vehicle Pair'].copy()
            speed_df['Speed_kmph'] = pd.to_numeric(speed_df['Speed_kmph'], errors='coerce')
            speed_df = speed_df.dropna(subset=['Speed_kmph'])

            bins = [0, 20, 40, 60, 80, 1000]
            bin_labels = ['0-20', '20-40', '40-60', '60-80', '80+']
            speed_df['Speed_Bin'] = pd.cut(speed_df['Speed_kmph'], bins=bins, labels=bin_labels, right=False)
            dist = speed_df['Speed_Bin'].value_counts().reindex(bin_labels, fill_value=0).to_dict()
            result["speed"] = {"labels": bin_labels, "data": [int(v) for v in dist.values()]}

            type_counts = speed_df['Vehicle_Type'].value_counts().to_dict()
            result["types"] = {"labels": list(type_counts.keys()), "data": [int(v) for v in type_counts.values()]}

        if 'Status' in df.columns:
            danger_count = int(df[df['Status'].str.contains('DANGER', na=False)].shape[0])
            overspeed_count = int(df[df['Status'] == 'DANGER'].shape[0])
            result["danger"] = {"labels": ["Too Close", "Overspeed"], "data": [danger_count, overspeed_count]}

        # Lane distribution from Frame Logs
        if 'Lane' in df.columns:
            lane_df = df[df['Vehicle_Type'] != 'Vehicle Pair']
            lane_counts = lane_df['Lane'].value_counts().to_dict()
            result["lanes"] = {"labels": list(lane_counts.keys()), "data": [int(v) for v in lane_counts.values()]}

        # Risk distribution from Vehicle Summary sheet
        try:
            df_summary = pd.read_excel(excel_path, sheet_name='Vehicle Summary')
            if not df_summary.empty:
                # Update total vehicles from the summary sheet (the most reliable source after filtering)
                result["total_vehicles"] = len(df_summary)
                
                if 'Accident_Risk' in df_summary.columns:
                    risk_counts = df_summary['Accident_Risk'].value_counts()
                    result["risk"] = {
                        "labels": ["Low", "Medium", "High"],
                        "data": [
                            int(risk_counts.get("Low", 0)),
                            int(risk_counts.get("Medium", 0)),
                            int(risk_counts.get("High", 0))
                        ]
                    }
        except Exception as e:
            print(f"Error reading Vehicle Summary: {e}")

    except Exception as e:
        print("Error processing excel:", e)

    return result


# DOWNLOAD VIDEO
@app.route("/download-video")
@app.route("/download-video/<int:history_id>")
def download_video(history_id=None):
    if history_id:
        video_path = os.path.join("static", "outputs", str(history_id), "output.mp4")
    else:
        video_path = os.path.join("static", "outputs", "output.mp4")

    return send_file(
        video_path,
        as_attachment=True,
        download_name="traffic_output.mp4"
    )


# DOWNLOAD EXCEL
@app.route("/download-excel")
@app.route("/download-excel/<int:history_id>")
def download_excel(history_id=None):
    if history_id:
        excel_path = os.path.join("static", "outputs", str(history_id), "traffic_logs.xlsx")
    else:
        excel_path = os.path.join("static", "outputs", "traffic_logs.xlsx")

    return send_file(
        excel_path,
        as_attachment=True,
        download_name="traffic_logs.xlsx"
    )


if __name__ == "__main__":
    app.run(debug=True)
