from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_action():
    video = request.files.get('video')
    if not video:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    video_path = os.path.join('uploads', video.filename)
    video.save(video_path)

    output_path = os.path.join('outputs', 'output.mp4')

    checkpoint_path = 'weights/best_top1_acc_epoch_21.pth'
    det_score_thr = '0.4'

    command = [
        'python', '../demo/demo_skeleton_frame.py',
        video_path,
        output_path,
        '--checkpoint', checkpoint_path,
        '--det-score-thr', det_score_thr
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)

        os.remove(video_path)
        output_lines = result.stdout.strip().split("\n")
        most_common_action = output_lines[-1].replace("action: ", "")

        return jsonify({'action': most_common_action}), 200
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        os.remove(video_path)
        return jsonify({'error': 'Failed to process video', 'details': e.stderr}), 500

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    app.run(host='0.0.0.0', port=443)