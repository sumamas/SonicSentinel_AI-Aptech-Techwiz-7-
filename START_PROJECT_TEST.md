# SonicSentinel: project ke saath 10-class model test

Is integrated release ko NEW folder mein extract karein. Purana project backup rakhein.
Model already trained hai; Colab ya train_models.bat dobara chalane ki zaroorat nahi.
Purani .venv copy na karein. Python 3.11 use karein.

## Asaan setup

1. XAMPP mein MySQL Start karein. Apache is Flask app ke liye zaroori nahi.
2. Project ki .env mein apni DB settings check karein. Default port 3307 aur database sonicsentinel hai.
3. setup_project_test.bat double-click karein. Internet par packages install honge; TensorFlow bara package hai.
4. READY aane ke baad start_project.bat double-click karein.
5. Browser mein http://127.0.0.1:5000 kholein. Terminal khula rakhein.

Is UI release mein event_analyses aur event_actions ki do nayi tables hain.
Purana database ho tab bhi setup ya python scripts\init_db.py ek baar run karein.
Database setup CREATE IF NOT EXISTS/create_all use karta hai, existing rows delete nahi karta.
Existing compatible database ho to apna login use karein. Naye database par Register se account banayein.
This is a local development setup, not a hosted deployment.

## CMD se setup

Jis folder mein app.py aur requirements.txt hain, wahin CMD kholein:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts\init_db.py
python scripts\check_project_model.py --database
python -m flask --app app run --host 127.0.0.1 --port 5000
```

## Upload test

- Login ke baad http://127.0.0.1:5000/audio/upload par jayein.
- Pehle clear 1-3 second WAV ya MP3 select karein; WAV, MP3, FLAC, OGG, M4A supported hain.
- Analyze karein. Python model CUSTOM_CNN, class, confidence aur top 3 predictions dikhni chahiye.
- Pehli request model/feature initialization ki wajah se slow ho sakti hai.
- Event History mein uploaded result save hota hai.
- Uncertain / Manual Review ka matlab prediction available hai, lekin confidence, class margin,
  quality ya clip length review rule match hua. Yeh missing-model error nahi.
- 3 seconds se lambi clip par highest-energy 3-second window classify hoti hai aur review flag lagta hai.
  60 seconds se lambi recording ko pehle chhote clips mein split karein.

## Live microphone test

- http://127.0.0.1:5000/workspace/live-monitoring kholein.
- Browser microphone permission Allow karein aur Start Monitoring use karein.
- Real microphone test ke liye localhost URL hi use karein.
- Dusre device se clear sample play karke Python result dekhein. Speaker playback aur room noise
  uploaded audio se different results de sakte hain. Pehle upload test verify karein.
- Live windows temporary hain; har window Event History mein save nahi hoti.
- GTM not connected rehna expected hai: is export mein Python model connected hai.

## Optional file test in terminal

```bat
python scripts\check_project_model.py --audio "C:\path\sample.wav"
python scripts\test_audio_picker.py
```

First command selected CNN check karta hai. Picker teen Python models compare karta hai.

## Kya connect hua

Purane root model selection mein 9-class SVM tha; latest_10class_model alag folder mein tha.
Flask root src/models use karta hai. Ab root mein exported 10-class CNN selection, RF/SVM,
matching v5 preprocessing, FFmpeg decoder, manifests aur package pins integrated hain.
Hashed preprocessing source files byte-for-byte preserved hain; authenticity checks retained hain.
Duplicate latest_10class_model folder ZIP mein repeat nahi kiya; active export root mein hai.
Existing frontend, authentication, routes, database settings aur uploaded audio retained hain.

## Result limits

Integration se accuracy improve hone ka claim nahi: selected CNN test accuracy 74.96%, macro F1 0.6727.
Panic Scream recall 34.48%; provisional review gate ke baad test recall 0% tha.
Use this release to test project wiring and behavior. Full SRS verified remains false.
Confidence score model ki overall accuracy nahi hoti. Apne fresh labeled samples se predictions compare karein.

## Common errors

- Database connection refused: MySQL Start karein; .env DB_PORT=3307 aur credentials check karein.
- No module / package mismatch: correct .venv activate karke requirements.txt install karein.
- Preprocessing/classes/dependencies mismatch: is ZIP ke src, models aur requirements together use karein.
  Files manually mix ya hash checks disable na karein.
- FFmpeg missing: requirements.txt imageio-ffmpeg install karta hai; install log check karein.
- Prediction unavailable: UI ka complete error aur terminal output share karein.
- Port 5000 busy: purana Flask terminal Ctrl+C se band karein.


## Naye UI ka quick test

1. Account menu se Workspace Overview kholein.
2. Upload Audio mein recording analyze karein; saved detail page par playback, top-three/all-ten scores aur actual waveform/spectrogram check karein.
3. Event History filters, Batch Upload aur Model Evaluation kholein.
4. Review/alert/settings controls ke liye apne registered account ko local administrator script se role dein (README.md mein command hai). Normal users ke liye privileged controls disabled/hidden rehna expected hai.
5. Teachable Machine Deferred/Not connected rehna expected hai.

Detailed scope: SRS_UI_ALIGNMENT.md. This is a modernized prototype; full SRS verification and critical recall targets are not yet passed.
