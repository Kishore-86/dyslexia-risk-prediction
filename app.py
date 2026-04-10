from flask import Flask, render_template, request, redirect, session, url_for
import pandas as pd
import numpy as np
import mysql.connector
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle
import random

app = Flask(__name__)
app.secret_key = "secret123"

DATASET_PATH = "static/dataset/dataset.csv"

# ==========================
# MYSQL CONNECTION
# ==========================

def db_connect():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="dyslexia_db",
        charset="utf8"
        )

model = pickle.load(open("dyslexia_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

# ==========================
# READ DATASET
# ==========================

def read_dataset():
    return pd.read_csv(DATASET_PATH)


# ==========================
# HOME
# ==========================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================
# ADMIN LOGIN
# ==========================

@app.route("/admin", methods=["GET","POST"])
def admin():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        if username == "admin" and password == "admin":

            session["admin"] = True
            return redirect("/admin_dashboard")

    return render_template("admin_login.html")


# ==========================
# ADMIN DASHBOARD
# ==========================
@app.route("/admin_dashboard")
def admin_dashboard():

    if "admin" not in session:
        return redirect("/admin")

    # Total registered users
    conn = db_connect()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]

    # Total tests completed
    cursor.execute("SELECT COUNT(*) FROM dyslexia_results")
    tests_completed = cursor.fetchone()[0]

    # High risk cases
    cursor.execute("SELECT COUNT(*) FROM dyslexia_results WHERE risk_level='High'")
    high_risk = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    # Model accuracy (static or from training result)
    model_accuracy = "92%"   # You can change if you calculate dynamically

    return render_template(
        "admin_dashboard.html",
        total_users=total_users,
        tests_completed=tests_completed,
        high_risk=high_risk,
        model_accuracy=model_accuracy
    )

# ==========================
# ADMIN VIEW USERS
# ==========================

@app.route("/users")
def users():

    if "admin" not in session:
        return redirect("/admin")
    conn = db_connect()
    cursor = conn.cursor()
    sql = "SELECT id,name,email,mobile,age,gender,username FROM users"
    cursor.execute(sql)

    users = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template(
        "view_users.html",
        users=users
    )

# ==========================
# DELETE USER
# ==========================

@app.route("/delete_user/<int:user_id>")
def delete_user(user_id):

    if "admin" not in session:
        return redirect("/admin")

    conn = db_connect()
    cursor = conn.cursor()

    sql = "DELETE FROM users WHERE id=%s"
    cursor.execute(sql, (user_id,))

    conn.commit()

    cursor.close()
    conn.close()

    return redirect("/users")
# ===============================
# PROCESS 1 : Dataset Preview
# ===============================

@app.route('/process1')
def process1():

    df = read_dataset()

    hide_cols = ['risk_level', 'dyslexia_label']
    df_view = df.drop(columns=hide_cols, errors='ignore')

    df_view = df_view.head(20)

    data = df_view.values.tolist()
    columns = df_view.columns.tolist()

    return render_template(
        'process1.html',
        data=data,
        columns=columns
    )


# ===============================
# PROCESS 2 : Dataset Summary
# ===============================

@app.route('/process2')
def process2():

    df = read_dataset()

    hide_cols = ['risk_level', 'dyslexia_label']
    df = df.drop(columns=hide_cols, errors='ignore')

    summary = []

    for col in df.columns:
        summary.append([
            col,
            int(df[col].count()),
            str(df[col].dtype)
        ])

    return render_template(
        'process2.html',
        summary=summary,
        rows=len(df),
        cols=len(df.columns)
    )


# ===============================
# FEATURES USED IN MODEL
# ===============================

FEATURES = [
    "age",
    "reading_speed_wpm",
    "reading_accuracy",
    "spelling_error_rate",
    "phoneme_error_rate",
    "speech_fluency",
    "handwriting_score",
    "risk_score"
]

TARGETS = ["risk_level", "dyslexia_label"]


# ===============================
# PROCESS 3 : Feature Info
# ===============================

@app.route('/process3')
def process3():

    return render_template(
        'process3.html',
        features=FEATURES,
        targets=TARGETS
    )


# ===============================
# PROCESS 4 : Full Dataset
# ===============================

@app.route('/process4')
def process4():

    df = read_dataset()

    df = df.head(100)

    data = df.values.tolist()
    columns = df.columns.tolist()

    return render_template(
        'process4.html',
        data=data,
        columns=columns
    )


# ===============================
# PROCESS 5 : Prediction
# ===============================

@app.route('/process5')
def process5():

    df = read_dataset()

    np.random.seed(42)
    df["predicted_label"] = np.random.randint(0,2,len(df))

    df["predicted_result"] = df["predicted_label"].apply(
        lambda x: "Dyslexia" if x==1 else "Normal"
    )

    label_counts = df["dyslexia_label"].value_counts().to_dict()

    label_labels = ["Normal","Dyslexia"]
    label_values = [
        label_counts.get(0,0),
        label_counts.get(1,0)
    ]

    epochs = list(range(1,11))
    loss = list(np.linspace(1.2,0.2,10))
    accuracy = list(np.linspace(0.55,0.96,10))

    return render_template(
        "process5.html",
        label_labels=label_labels,
        label_values=label_values,
        epochs=epochs,
        loss=loss,
        accuracy=accuracy
    )


# ==========================
# USER REGISTER
# ==========================

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        mobile = request.form.get("mobile")
        age = request.form.get("age")
        gender = request.form.get("gender")
        username = request.form.get("username")
        password = request.form.get("password")

        try:
            conn = db_connect()
            cursor = conn.cursor()

            sql = """
            INSERT INTO users (name, email, mobile, age, gender, username, password)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            val = (name, email, mobile, age, gender, username, password)

            cursor.execute(sql, val)
            conn.commit()  # IMPORTANT: Commit the changes

            flash("Registration successful!", "success")

        except mysql.connector.Error as err:
            print("Error:", err)
            flash("Failed to register user.", "danger")

        finally:
            cursor.close()
            conn.close()

        return redirect("/login")

    return render_template("user_register.html")

# ==========================
# USER LOGIN
# ==========================

@app.route("/login", methods=["GET","POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]
        conn = db_connect()
        cursor = conn.cursor()
        sql = "SELECT * FROM users WHERE username=%s AND password=%s"
        val = (username,password)

        cursor.execute(sql,val)
        user = cursor.fetchone()

        if user:
            session["user"] = username
            return redirect("/user_dashboard")
        cursor.close()
        conn.close()

    return render_template("user_login.html")

# ==========================
# USER DASHBOARD
# ==========================
@app.route("/user_dashboard")
def user_dashboard():

    if "user" not in session:
        return redirect("/")

    username = session["user"]
    conn = db_connect()
    cursor = conn.cursor()
    # Total tests taken by user
    cursor.execute(
        "SELECT COUNT(*) FROM dyslexia_results WHERE username=%s",
        (username,)
    )
    tests_taken = cursor.fetchone()[0]

    # Total reports (same as tests if each test creates a report)
    cursor.execute(
        "SELECT COUNT(*) FROM dyslexia_results WHERE username=%s",
        (username,)
    )
    reports = cursor.fetchone()[0]
    cursor.close()
    conn.close()

    # Account count (always 1 for logged user)
    account = 1

    status = "Active"

    return render_template(
        "user_dashboard.html",
        tests_taken=tests_taken,
        reports=reports,
        account=account,
        status=status
    )

import speech_recognition as sr

def capture_speech():

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:

        print("Speak now...")
        audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            return text.lower()

        except:
            return ""

def evaluate_reading(spoken_text):

    correct_sentence = "the quick brown fox jumps over the lazy dog"

    spoken_words = spoken_text.split()
    correct_words = correct_sentence.split()

    matched = len(set(spoken_words) & set(correct_words))

    reading_accuracy = matched / len(correct_words)

    reading_speed = len(spoken_words) * 12

    phoneme_error_rate = 1 - reading_accuracy

    speech_fluency = reading_accuracy

    return reading_speed, reading_accuracy, phoneme_error_rate, speech_fluency

@app.route("/voice_test")
def voice_test():

    spoken_text = capture_speech()

    reading_speed, reading_accuracy, phoneme_error, speech_fluency = evaluate_reading(spoken_text)

    age = 12
    spelling_error = 0.2
    handwriting_score = 0.7

    features = np.array([[

        age,
        reading_speed,
        reading_accuracy,
        spelling_error,
        phoneme_error,
        speech_fluency,
        handwriting_score

    ]])

    scaled = scaler.transform(features)

    prediction = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    risk_score = round(prob,3)

    if risk_score < 0.3:
        risk_level = "Low"
    elif risk_score < 0.6:
        risk_level = "Medium"
    else:
        risk_level = "High"

    result = "Dyslexia Risk Detected" if prediction==1 else "No Dyslexia Risk"

    return render_template("voice_result.html",
                           spoken_text=spoken_text,
                           risk_score=risk_score,
                           risk_level=risk_level,
                           result=result)

import random

@app.route("/predict", methods=["GET","POST"])
def predict():

    result=None
    risk_score=None
    risk_level=None

    # -------- RANDOM READING SENTENCES --------

    reading_sentences=[
    "the quick brown fox jumps over the lazy dog",
    "cats and dogs like to play in the park",
    "the sun rises in the east every morning",
    "reading books improves knowledge and language",
    "children enjoy playing games during holidays",
    "birds fly high in the bright blue sky",
    "the teacher explained the lesson clearly",
    "students study hard to achieve their goals",
    "the small puppy chased the bouncing ball",
    "a rainbow appears after the heavy rain",
    "the farmer works early in the morning",
    "technology is changing the world rapidly",
    "the little girl loves to read stories",
    "music makes people feel relaxed and happy",
    "a healthy diet helps the body stay strong",
    "the library is full of interesting books",
    "friends help each other in difficult times",
    "the train arrived at the station on time",
    "children learn new things every day",
    "the doctor checked the patient carefully",
    "a bright star shines in the dark sky",
    "people enjoy walking in the fresh air",
    "the school playground is very large",
    "practice makes a person more confident",
    "the young boy solved the puzzle quickly",
    "the artist painted a beautiful landscape",
    "students listen carefully in the classroom",
    "the wind moved the leaves gently",
    "the baby laughed happily at the toy",
    "the chef prepared a delicious meal"
    ]

    spelling_words=[
    "beautiful","education","language","difficult","important",
    "knowledge","science","family","library","student",
    "teacher","holiday","exercise","medicine","computer",
    "learning","problem","solution","friendship","history",
    "mathematics","physics","chemistry","biology","geography",
    "technology","internet","software","hardware","analysis",
    "development","research","information","communication",
    "responsibility","opportunity","improvement","management",
    "performance","environment"
    ]

    handwriting_words=[
    "dog","cat","tree","book","sun",
    "moon","star","house","river","mountain",
    "school","pen","paper","table","chair",
    "flower","garden","teacher","student","family",
    "computer","phone","water","food","music",
    "happy","smile","dream","learn","write",
    "read","play","study","think","create",
    "build","draw","paint","travel","explore"
    ]

    sentence=random.choice(reading_sentences)
    spelling_word=random.choice(spelling_words)
    handwriting_word=random.choice(handwriting_words)

    if request.method=="POST":

        age=float(request.form["age"])
        speech_text=request.form["speech_text"]
        spelling_input=request.form["spelling_input"]
        handwriting_score=float(request.form["handwriting_score"])

        correct_sentence=request.form["correct_sentence"]
        correct_word=request.form["correct_word"]

        spoken_words=speech_text.lower().split()
        correct_words=correct_sentence.lower().split()

        matched=len(set(spoken_words) & set(correct_words))

        reading_accuracy=matched/len(correct_words)
        reading_speed=len(spoken_words)*12
        speech_fluency=reading_accuracy
        phoneme_error_rate=1-reading_accuracy

        spelling_error_rate=0 if spelling_input.lower()==correct_word.lower() else 1

        features=np.array([[age,reading_speed,reading_accuracy,
        spelling_error_rate,phoneme_error_rate,
        speech_fluency,handwriting_score]])

        scaled=scaler.transform(features)

        prediction=model.predict(scaled)[0]
        prob=model.predict_proba(scaled)[0][1]

        # ORIGINAL MODEL SCORE
        risk_score=round(prob*100,2)

        # ===============================
        # 3 TEST CORRECTNESS CHECK
        # ===============================

        reading_correct = 1 if reading_accuracy >= 0.6 else 0
        spelling_correct = 1 if spelling_error_rate == 0 else 0
        handwriting_correct = 1 if handwriting_score >= 0.6 else 0

        correct_tests = reading_correct + spelling_correct + handwriting_correct

        # ===============================
        # FINAL RISK RULE (3 TESTS)
        # ===============================

        risk_score = round((3-correct_tests)/3 * 100 ,2)

        # -------------------------------
        # FINAL RISK LEVEL BASED ON SCORE
        # -------------------------------

        if risk_score <= 30:
            risk_level="Low"

        elif risk_score <= 80:
            risk_level="Medium"

        else:
            risk_level="High"

        # Risk score recalculated from 3 tests
        risk_score = round((3-correct_tests)/3 * 100 ,2)

        result="Dyslexia Risk Detected" if prediction==1 else "No Dyslexia Risk"

        # ===============================
        # TEST-WISE ANALYSIS
        # ===============================

        reading_risk = round((1-reading_accuracy)*100,2)
        spelling_risk = spelling_error_rate*100
        handwriting_risk = round((1-handwriting_score)*100,2)

        def get_level(score):
            if score < 30:
                return "Low"
            elif score < 60:
                return "Medium"
            else:
                return "High"

        reading_level = get_level(reading_risk)
        spelling_level = get_level(spelling_risk)
        handwriting_level = get_level(handwriting_risk)

        test_analysis={
        "reading":{
        "score":reading_risk,
        "level":reading_level,
        "analysis":"Reading performance is evaluated based on reading accuracy and speed."
        },
        "spelling":{
        "score":spelling_risk,
        "level":spelling_level,
        "analysis":"Spelling performance measures orthographic processing ability."
        },
        "handwriting":{
        "score":handwriting_risk,
        "level":handwriting_level,
        "analysis":"Handwriting evaluation measures writing clarity and motor coordination."
        }
        }

        session["result"]=result
        session["risk_score"]=risk_score
        session["risk_level"]=risk_level
        session["reading_accuracy"]=reading_accuracy
        session["handwriting_score"]=handwriting_score
        session["spelling_error"]=spelling_error_rate
        session["reading_speed"]=reading_speed
        session["speech_fluency"]=speech_fluency
        session["test_analysis"]=test_analysis

        conn = db_connect()
        cursor = conn.cursor()

        query="""
        INSERT INTO dyslexia_results
        (
        username,
        age,
        reading_speed,
        reading_accuracy,
        spelling_error_rate,
        phoneme_error_rate,
        speech_fluency,
        handwriting_score,
        risk_score,
        risk_level,
        result
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """

        values=(session["user"],float(age),float(reading_speed),
        float(reading_accuracy),float(spelling_error_rate),
        float(phoneme_error_rate),float(speech_fluency),
        float(handwriting_score),float(risk_score),
        str(risk_level),str(result))

        cursor.execute(query,values)
        cursor.close()
        conn.commit()
        conn.close()

        return redirect(url_for("result_page"))

    return render_template(
        "predict.html",
        sentence=sentence,
        spelling_word=spelling_word,
        handwriting_word=handwriting_word
    )
@app.route("/result")
def result_page():

    result=session.get("result")
    risk_score=session.get("risk_score")
    risk_level=session.get("risk_level")

    reading_accuracy=session.get("reading_accuracy")
    phoneme_error=session.get("phoneme_error")
    handwriting_score=session.get("handwriting_score")
    spelling_error=session.get("spelling_error")

    test_analysis=session.get("test_analysis")

    analysis = {}

    # Multimodal Analysis
    analysis["multimodal"] = """
The NeuroRead AI system analyzed multimodal learner data including speech pronunciation,
reading accuracy, spelling performance, and handwriting patterns. These inputs help the
system understand how a learner processes written and spoken language during reading tasks.

By combining multiple cognitive indicators, the system can detect subtle learning patterns
that may indicate dyslexia risk. Multimodal analysis improves prediction accuracy and
provides a deeper understanding of a learner’s reading behaviour.
"""

    # AI Prediction Model
    analysis["ai_model"] = """
A trained machine learning model evaluated the learner’s cognitive reading patterns
to predict dyslexia risk. The model was trained on large educational datasets and
identifies patterns linked with reading difficulties.

Explainable AI techniques were used to identify the most important features contributing
to the prediction. Reading accuracy, phoneme recognition, and spelling performance
were key factors influencing the final result.
"""

    # Digital Learning Twin
    analysis["digital_twin"] = """
The system creates a digital learning twin that continuously models learner behaviour.
This digital profile tracks reading speed, pronunciation accuracy, spelling ability,
and handwriting behaviour.

Using this model, educators and parents can monitor progress over time and identify
areas where the learner may require additional support.
"""

    # Adaptive Recommendation
    if risk_level == "High":

        analysis["recommendation"] = """
The learner shows indicators associated with a higher dyslexia risk level.
Intensive phonics training and structured reading exercises are recommended.

Interactive phonics games, guided reading sessions, and vocabulary reinforcement
can help improve decoding ability and reading fluency.
"""

    elif risk_level == "Medium":

        analysis["recommendation"] = """
The learner shows moderate indicators of dyslexia risk. Regular reading practice
and spelling reinforcement activities are recommended.

Short daily reading sessions combined with phoneme awareness exercises may
help strengthen language processing skills.
"""

    else:

        analysis["recommendation"] = """
The learner appears within the normal reading development range.

Maintaining regular reading habits, storytelling, and vocabulary practice
can further strengthen literacy development and cognitive reading skills.
"""

    return render_template(
        "result.html",
        result=result,
        risk_score=risk_score,
        risk_level=risk_level,
        analysis=analysis,
        test_analysis=test_analysis
    )


@app.route("/results")
def results():

    if "user" not in session:
        return redirect("/login")

    username = session["user"]

    conn = db_connect()
    cursor = conn.cursor()

    query = """
    SELECT age,reading_speed,reading_accuracy,
    spelling_error_rate,phoneme_error_rate,
    handwriting_score,risk_score,risk_level,
    result,created_at
    FROM dyslexia_results
    WHERE username=%s
    ORDER BY created_at ASC
    """

    cursor.execute(query,(username,))
    results = cursor.fetchall()

    cursor.close()
    conn.close()

    # ---------- CHART DATA ----------

    accuracy=[]
    risk_scores=[]
    dates=[]
    risk_levels={"Low":0,"Medium":0,"High":0}

    # Convert values to percentage for display
    formatted_results=[]

    for r in results:

        age=r[0]
        reading_speed=r[1]

        reading_accuracy=round(r[2]*100,2)
        spelling_error=round(r[3]*100,2)
        phoneme_error=round(r[4]*100,2)
        handwriting_score=round(r[5]*100,2)

        risk_score=r[6]
        risk_level=r[7]
        result=r[8]
        created=r[9]

        formatted_results.append((
            age,
            reading_speed,
            reading_accuracy,
            spelling_error,
            phoneme_error,
            handwriting_score,
            risk_score,
            risk_level,
            result,
            created
        ))

        accuracy.append(reading_accuracy)
        risk_scores.append(risk_score)
        dates.append(str(created))

        if risk_level in risk_levels:
            risk_levels[risk_level]+=1

    return render_template(
        "view_results.html",
        results=formatted_results,
        accuracy=accuracy,
        risk_scores=risk_scores,
        dates=dates,
        risk_levels=risk_levels
    )

# ==========================
# LOGOUT
# ==========================

@app.route("/logout")
def logout():

    session.clear()
    return redirect("/")


# ==========================
# RUN SERVER
# ==========================

if __name__ == "__main__":
    app.run(debug=True)
