import streamlit as st
import cv2
import numpy as np
import mysql.connector
from PIL import Image
import os
import smtplib
from email.message import EmailMessage
from datetime import datetime
import shutil
import time
import uuid


# Initialize session state variables
if 'passkey_attempts' not in st.session_state:
    st.session_state.passkey_attempts = 0

if 'access_granted' not in st.session_state:
    st.session_state.access_granted = False

if 'show_main_options' not in st.session_state:
    st.session_state.show_main_options = False

if 'dataset_info' not in st.session_state:
    st.session_state.dataset_info = {'name': '', 'age': '', 'address': ''}

if 'crud_operation_verified' not in st.session_state:
    st.session_state.crud_operation_verified = False

if 'operation_type' not in st.session_state:
    st.session_state.operation_type = None

if 'selected_user_id' not in st.session_state:
    st.session_state.selected_user_id = None

if 'user_to_edit' not in st.session_state:
    st.session_state.user_to_edit = {'id': None, 'name': '', 'age': '', 'address': ''}

if 'authenticated_user' not in st.session_state:
    st.session_state.authenticated_user = None

if 'auth_method' not in st.session_state:
    st.session_state.auth_method = None

# Create data directory if it doesn't exist
data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Load Haar Cascade for face detection
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascade_path)

if faceCascade.empty():
    st.error("Error: Haar Cascade XML file could not be loaded.")
    st.stop()

# Load LBPH Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load classifier if it exists
classifier_exists = False
try:
    recognizer.read("classifier.xml")
    classifier_exists = True
except Exception as e:
    st.warning("Classifier not found. Train the model before detecting faces.")

# Database connection function
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="",
            database="Authorized_users"
        )
        return conn
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None

# Initialize database if it doesn't exist
def init_database():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd=""
        )
        cursor = conn.cursor()
        
        # Create database if it doesn't exist
        cursor.execute("CREATE DATABASE IF NOT EXISTS Authorized_users")
        cursor.execute("USE Authorized_users")
        
        # Check if table exists
        cursor.execute("SHOW TABLES LIKE 'my_table'")
        table_exists = cursor.fetchone()
        
        # Create table if it doesn't exist with explicit AUTO_INCREMENT definition
        if not table_exists:
            cursor.execute("""
            CREATE TABLE my_table (
                id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                age VARCHAR(50),
                address TEXT
            ) ENGINE=InnoDB AUTO_INCREMENT=1
            """)
        
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Database initialization error: {e}")
        return False
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

# Email Alert Function
def send_email_alert(person_name, confidence, img_path):
    try:
        email_sender = "samyak.1403@gmail.com"
        email_password = "jpcngoqvqeapstnz"  # Consider using environment variables for this
        email_receiver = "deepanshiajmera1304@gmail.com"

        subject = "🔔 Face Detection Alert"
        body = f"Person Detected: {person_name}\nConfidence Level: {confidence}%\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        msg = EmailMessage()
        msg['From'] = email_sender
        msg['To'] = email_receiver
        msg['Subject'] = subject
        msg.set_content(body)

        with open(img_path, 'rb') as f:
            img_data = f.read()
            msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename='detected_face.jpg')

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(email_sender, email_password)
            server.send_message(msg)

        st.info(f"Email alert sent for {person_name}")
        return True
    except Exception as e:
        st.error(f"Email sending error: {e}")
        return False

# Detect Face Function
def detect_face():
    if not classifier_exists:
        st.error("Face recognition model not trained yet. Please add users and train classifier first.")
        return False
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Cannot access the webcam. Check your camera permissions!")
        return False

    stframe = st.empty()
    status_placeholder = st.empty()
    start_time = datetime.now()
    
    status_placeholder.info("Detecting face... Look at the camera")
    
    # Track if we've found a valid face
    face_detected = False
    person_name = None
    best_confidence = 0

    while (datetime.now() - start_time).seconds < 20:  # Maximum 20 seconds timeout
        ret, img = cap.read()
        if not ret:
            status_placeholder.error("Failed to read frame from webcam.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            face_roi = gray[y:y+h, x:x+w]
            
            try:
                id, pred = recognizer.predict(face_roi)
                confidence = int(100 * (1 - pred / 300))
                
                conn = get_db_connection()
                if not conn:
                    status_placeholder.error("Could not connect to database")
                    continue
                    
                cursor = conn.cursor()
                cursor.execute(f"SELECT name FROM my_table WHERE id={id}")
                user_data = cursor.fetchone()
                cursor.close()
                conn.close()

                if user_data:
                    current_name = user_data[0]
                    cv2.putText(img, f"{current_name} ({confidence}%)", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    
                    # Track the best confidence we've seen
                    if confidence > best_confidence:
                        best_confidence = confidence
                        person_name = current_name
                        
                    # Only set face_detected if we exceed the threshold
                    if confidence > 70:
                        face_detected = True
                else:
                    cv2.putText(img, f"Unknown ({confidence}%)", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    
                    # Save unknown face and send alert after a few seconds
                    if (datetime.now() - start_time).seconds > 5:
                        img_path = os.path.join(data_dir, "unknown_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".jpg")
                        cv2.imwrite(img_path, img)
                        send_email_alert("Unknown", confidence, img_path)
                        status_placeholder.error("Unauthorized Person Detected. Alert Sent.")

            except Exception as e:
                status_placeholder.error(f"Recognition Error: {e}")

        # Convert color for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        stframe.image(img_rgb, caption="Face Detection", use_column_width=True)
        
        # Check if we've found a valid face with high confidence
        if face_detected:
            # Wait a bit longer to ensure we have a stable reading
            if (datetime.now() - start_time).seconds > 3:
                break

    cap.release()
    
    # Only grant access if a valid face was detected
    if face_detected and best_confidence > 70:
        st.session_state.access_granted = True
        # Store the authenticated user's name
        st.session_state.authenticated_user = person_name
        # Set authentication method to face recognition
        st.session_state.auth_method = "face"
        # With face recognition, CRUD operations are pre-verified
        st.session_state.crud_operation_verified = True
        status_placeholder.success(f"Access Granted to {person_name} with {best_confidence}% confidence")
        return True
    else:
        if best_confidence > 0:
            status_placeholder.warning(f"Face detected but confidence too low ({best_confidence}%). Access denied.")
        else:
            status_placeholder.warning("No recognized face detected. Access denied.")
        return False
        
# Dataset Generation Function
def generate_dataset(user_id):
    name = st.session_state.dataset_info['name']
    
    # Create directory for user images
    person_dir = os.path.join(data_dir, f"user_{user_id}")
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Cannot access the webcam. Check your camera permissions!")
        return False
        
    counter = 0
    total_samples = 30
    progress_bar = st.progress(0)
    status_text = st.empty()
    frame_display = st.empty()
    
    status_text.info(f"Capturing face samples for {name}. Please look at the camera and move your head slightly.")
    
    while counter < total_samples:
        ret, img = cap.read()
        if not ret:
            st.error("Error capturing image from webcam.")
            break
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) > 0:  # Only save if a face is detected
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Save only the face region
                face_img = gray[y:y+h, x:x+w]
                img_path = os.path.join(person_dir, f"{counter}.jpg")
                cv2.imwrite(img_path, face_img)
                
                counter += 1
                progress_bar.progress(counter / total_samples)
                status_text.info(f"Capturing sample {counter}/{total_samples}")
                
                if counter >= total_samples:
                    break
                
        # Display the frame
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_display.image(img_rgb, caption="Capturing face data", use_column_width=True)
            
    cap.release()
    
    if counter >= total_samples:
        status_text.success(f"Dataset generated successfully with {counter} samples")
        return True
    else:
        status_text.error(f"Only captured {counter}/{total_samples} samples. Try again.")
        return False

# Train Classifier Function
def train_classifier():
    global classifier_exists  # Properly declare global variable
    
    status = st.empty()
    status.info("Training classifier...")
    
    face_samples = []
    face_ids = []
    
    # Connect to database to get user IDs
    conn = get_db_connection()
    if not conn:
        status.error("Could not connect to database for training")
        return False
        
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM my_table")
    db_users = cursor.fetchall()
    cursor.close()
    conn.close()
    
    # Exit if no users
    if not db_users:
        status.error("No users found in database. Add users first.")
        return False
    
    # Load all face samples
    for user_id, name in db_users:
        user_dir = os.path.join(data_dir, f"user_{user_id}")
        if not os.path.exists(user_dir):
            status.warning(f"No data directory found for {name} (ID: {user_id})")
            continue
            
        status.info(f"Loading training data for {name}...")
        
        for img_file in os.listdir(user_dir):
            if not img_file.endswith('.jpg'):
                continue
                
            img_path = os.path.join(user_dir, img_file)
            face_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if face_img is not None:
                face_samples.append(face_img)
                face_ids.append(user_id)
    
    if not face_samples:
        status.error("No face samples found. Generate datasets first.")
        return False
    
    # Train the recognizer
    status.info(f"Training with {len(face_samples)} images...")
    recognizer.train(face_samples, np.array(face_ids))
    
    # Save trained model
    recognizer.write("classifier.xml")
    classifier_exists = True  # Set after successful training
    status.success("Classifier trained successfully!")
    return True

# Add user to database
def add_user_to_db():
    name = st.session_state.dataset_info['name']
    age = st.session_state.dataset_info['age']
    address = st.session_state.dataset_info['address']
    
    if not name or not age or not address:
        st.error("Please provide complete details before submitting.")
        return None
    
    conn = get_db_connection()
    if not conn:
        return None
        
    try:
        cursor = conn.cursor()
        
        # Find the smallest available ID (gap)
        cursor.execute("SELECT MIN(t1.id + 1) AS next_id FROM my_table t1 LEFT JOIN my_table t2 ON t1.id + 1 = t2.id WHERE t2.id IS NULL")
        next_id = cursor.fetchone()[0]
        
        # If no gaps, use the next sequential ID
        if next_id is None:
            cursor.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM my_table")
            next_id = cursor.fetchone()[0]
        
        # Insert the new user with the calculated ID
        query = "INSERT INTO my_table (id, name, age, address) VALUES (%s, %s, %s, %s)"
        values = (next_id, name, age, address)
        cursor.execute(query, values)
        conn.commit()
        
        st.success(f"User {name} added to database with ID: {next_id}")
        return next_id
    except Exception as e:
        st.error(f"Error adding user to database: {e}")
        return None
    finally:
        if 'cursor' in locals():
            cursor.close()
        conn.close()


# Update user in database
def update_user_in_db(user_id, name, age, address):
    conn = get_db_connection()
    if not conn:
        return False
        
    try:
        cursor = conn.cursor()
        query = "UPDATE my_table SET name = %s, age = %s, address = %s WHERE id = %s"
        values = (name, age, address, user_id)
        cursor.execute(query, values)
        conn.commit()
        
        if cursor.rowcount > 0:
            st.success(f"User {name} (ID: {user_id}) updated successfully")
            return True
        else:
            st.warning(f"No changes made to user {name} (ID: {user_id})")
            return False
    except Exception as e:
        st.error(f"Error updating user: {e}")
        return False
    finally:
        if 'cursor' in locals():
            cursor.close()
        conn.close()


# Delete user from database and remove their images
def delete_user_from_db(user_id):
    # First get user info for confirmation
    conn = get_db_connection()
    if not conn:
        return False
        
    try:
        cursor = conn.cursor()
        
        # Get user name for confirmation
        cursor.execute("SELECT name FROM my_table WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        
        if not user:
            st.error(f"User with ID {user_id} not found")
            return False
            
        user_name = user[0]
        
        # Delete user from database
        cursor.execute("DELETE FROM my_table WHERE id = %s", (user_id,))
        conn.commit()
        
        # Reorganize IDs to fill the gap
        cursor.execute("SET @count = 0")
        cursor.execute("UPDATE my_table SET id = @count:= @count + 1 ORDER BY id")
        cursor.execute("ALTER TABLE my_table AUTO_INCREMENT = 1")
        conn.commit()
        
        # Delete user's image directory
        user_dir = os.path.join(data_dir, f"user_{user_id}")
        if os.path.exists(user_dir):
            shutil.rmtree(user_dir)
            
        st.success(f"User {user_name} (ID: {user_id}) deleted successfully along with all face data")
        return True
    except Exception as e:
        st.error(f"Error deleting user: {e}")
        return False
    finally:
        if 'cursor' in locals():
            cursor.close()
        conn.close()


# Get user by ID
def get_user_by_id(user_id):
    conn = get_db_connection()
    if not conn:
        return None
        
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, name, age, address FROM my_table WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        return user
    except Exception as e:
        st.error(f"Error fetching user: {e}")
        return None
    finally:
        if 'cursor' in locals():
            cursor.close()
        conn.close()
# Verify passkey for CRUD operations
def verify_passkey_for_crud():
    st.subheader("Authentication Required")
    st.warning("Please enter the passkey to access admin functions")
    
    passkey = st.text_input("Enter Admin Passkey", type="password", key="crud_passkey")
    
    if st.button("Verify Passkey"):
        if passkey == "securepass123":
            st.session_state.crud_operation_verified = True
            st.success("Passkey verified. You can now perform any operation.")
            st.experimental_rerun()
        else:
            st.error("Incorrect passkey. Access denied.")

# Initialize database on app start
init_database()

# Main App
st.title("Face Recognition System")

# Add some CSS for animations
st.markdown("""
<style>
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

.fadeIn {
    animation: fadeIn 1s ease-in-out;
}

@keyframes slideIn {
    from { transform: translateX(-100%); }
    to { transform: translateX(0); }
}

.slideIn {
    animation: slideIn 0.5s ease-in-out;
}

.stButton>button {
    background-color: #4CAF50;
    color: white;
    padding: 10px 20px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}

.stButton>button:hover {
    background-color: #45a049;
}

.stTextInput>div>div>input {
    padding: 10px;
    border-radius: 5px;
    border: 1px solid #ccc;
    transition: border-color 0.3s ease;
}

.stTextInput>div>div>input:focus {
    border-color: #4CAF50;
    outline: none;
}

.stProgress>div>div>div {
    background-color: #4CAF50;
}

.stAlert {
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
}

.stAlert.success {
    background-color: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}

.stAlert.error {
    background-color: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
}

.stAlert.warning {
    background-color: #fff3cd;
    color: #856404;
    border: 1px solid #ffeeba;
}

.stAlert.info {
    background-color: #d1ecf1;
    color: #0c5460;
    border: 1px solid #bee5eb;
}
</style>
""", unsafe_allow_html=True)

if not st.session_state.access_granted:
    option = st.radio("Choose Authentication Method", ["Face Recognition", "Passkey"])

    if option == "Face Recognition":
        if st.button("Detect Face"):
            with st.spinner("Detecting face..."):
                detect_face()

    elif option == "Passkey":
        passkey = st.text_input("Enter Passkey", type="password")

        if st.button("Submit Passkey"):
            if passkey == "securepass123":
                st.session_state.access_granted = True
                # Set authenticated user as Admin for passkey login
                st.session_state.authenticated_user = "Admin"
                # Set authentication method to passkey
                st.session_state.auth_method = "passkey"
                # Note: CRUD operations still need verification
                st.session_state.crud_operation_verified = False
                st.success("Access Granted via Passkey")
            else:
                st.session_state.passkey_attempts += 1
                st.error(f"Incorrect Passkey. Access Denied. ({st.session_state.passkey_attempts}/3 attempts)")

            if st.session_state.passkey_attempts >= 3:
                st.error("Too many failed attempts. Switching to face recognition.")
                detect_face()

if st.session_state.access_granted:
    st.title("Authorized User Dashboard")
    
    # Display authenticated user name in a container above the tabs
    user_info_col1, user_info_col2 = st.columns([3, 1])
    with user_info_col1:
        if st.session_state.authenticated_user:
            st.info(f"Welcome, {st.session_state.authenticated_user}!")
    
    with user_info_col2:
        if st.button("Log Out"):
            st.session_state.access_granted = False
            st.session_state.show_main_options = False
            st.session_state.operation_type = None
            st.session_state.crud_operation_verified = False
            st.session_state.selected_user_id = None
            st.session_state.authenticated_user = None
            st.session_state.auth_method = None
            st.experimental_rerun()
    
    # Check if this is passkey authentication and verification is still needed
    if st.session_state.auth_method == "passkey" and not st.session_state.crud_operation_verified:
        verify_passkey_for_crud()
    else:
        # User is verified (either by face or passkey+verification), show all tabs and operations
        tabs = st.tabs(["Add User", "Train System", "User Management"])
        
        with tabs[0]:
            st.header("Add New User")
            
            st.session_state.dataset_info['name'] = st.text_input("Name")
            st.session_state.dataset_info['age'] = st.text_input("Age")
            st.session_state.dataset_info['address'] = st.text_area("Address")

            if st.button("Add User to System"):
                with st.spinner("Adding user..."):
                    user_id = add_user_to_db()
                    if user_id:
                        if generate_dataset(user_id):
                            st.success(f"User {st.session_state.dataset_info['name']} added successfully.")
                            st.info("Remember to train the classifier after adding users.")
        
        with tabs[1]:
            st.header("Train Recognition System")
            if st.button("Train Classifier"):
                with st.spinner("Training classifier..."):
                    train_classifier()
        
        with tabs[2]:
            st.header("User Management")
            
            # Create subtabs for different operations
            user_tabs = st.tabs(["View Users", "Edit User", "Delete User"])
            
            with user_tabs[0]:
                if st.button("View Registered Users", key="view_users"):
                    conn = get_db_connection()
                    if conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT id, name, age, address FROM my_table")
                        users = cursor.fetchall()
                        cursor.close()
                        conn.close()
                        
                        if users:
                            user_df = np.array(users)
                            st.table({
                                "ID": user_df[:, 0],
                                "Name": user_df[:, 1],
                                "Age": user_df[:, 2],
                                "Address": user_df[:, 3]
                            })
                        else:
                            st.info("No users registered in the system.")
            
            with user_tabs[1]:
                st.subheader("Edit User Information")
                
                conn = get_db_connection()
                if conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT id, name FROM my_table")
                    users = cursor.fetchall()
                    cursor.close()
                    conn.close()
                    
                    if users:
                        user_options = [f"{user[0]}: {user[1]}" for user in users]
                        selected_user = st.selectbox("Select user to edit:", user_options)
                        
                        # Extract user ID from selection
                        user_id = int(selected_user.split(":")[0])
                        
                        # Get user data for editing
                        user = get_user_by_id(user_id)
                        
                        if user:
                            # Pre-fill form with user data
                            new_name = st.text_input("Name", value=user['name'], key="edit_name")
                            new_age = st.text_input("Age", value=user['age'], key="edit_age")
                            new_address = st.text_area("Address", value=user['address'], key="edit_address")
                            
                            if st.button("Update User"):
                                with st.spinner("Updating user..."):
                                    if update_user_in_db(user_id, new_name, new_age, new_address):
                                        st.success(f"User {new_name} updated successfully")
                        else:
                            st.error(f"User data not found")
                    else:
                        st.info("No users registered in the system.")
            
            with user_tabs[2]:
                st.subheader("Delete User")
                
                conn = get_db_connection()
                if conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT id, name FROM my_table")
                    users = cursor.fetchall()
                    cursor.close()
                    conn.close()
                    
                    if users:
                        user_options = [f"{user[0]}: {user[1]}" for user in users]
                        selected_user = st.selectbox("Select user to delete:", user_options, key="delete_user")
                        
                        # Extract user ID from selection
                        user_id = int(selected_user.split(":")[0])
                        
                        st.warning(f"Are you sure you want to delete this user? This will permanently remove the user and all associated face data.")
                        
                        if st.button("Confirm Delete"):
                            with st.spinner("Deleting user..."):
                                if delete_user_from_db(user_id):
                                    st.success("User deleted. You should retrain the classifier.")
                                    
                                    # Suggest retraining
                                    if st.button("Retrain Classifier Now"):
                                        with st.spinner("Retraining classifier..."):
                                            train_classifier()
                    else:
                        st.info("No users registered in the system.")

# ////////////////////////////////////////////////MOST INTERACTIVE//////////////////////////////////////////////////////////////////////

# import streamlit as st
# import cv2
# import numpy as np
# import mysql.connector
# from PIL import Image
# import os
# import smtplib
# from email.message import EmailMessage
# from datetime import datetime
# from datetime import datetime, timedelta

# import shutil
# import time
# from streamlit_lottie import st_lottie
# import requests
# import json
# import base64

# # Set page config for better appearance
# st.set_page_config(
#     page_title="Secure Face Recognition System",
#     page_icon="🔒",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS to enhance UI
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         color: #2c3e50;
#         text-align: center;
#         margin-bottom: 1rem;
#         text-shadow: 1px 1px 2px #ccc;
#     }
#     .sub-header {
#         font-size: 1.8rem;
#         color: #34495e;
#         margin-top: 1rem;
#     }
#     .card {
#         background-color: #f8f9fa;
#         border-radius: 10px;
#         padding: 20px;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#         margin-bottom: 20px;
#     }
#     .success-text {
#         color: #28a745;
#         font-weight: bold;
#     }
#     .warning-text {
#         color: #ffc107;
#         font-weight: bold;
#     }
#     .error-text {
#         color: #dc3545;
#         font-weight: bold;
#     }
#     .stButton button {
#         background-color: #4e73df;
#         color: white;
#         font-weight: bold;
#         border-radius: 5px;
#         padding: 10px 20px;
#         border: none;
#         transition: all 0.3s ease;
#     }
#     .stButton button:hover {
#         background-color: #2e59d9;
#         box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
#     }
#     .logout-btn {
#         background-color: #dc3545 !important;
#     }
#     .logout-btn:hover {
#         background-color: #c82333 !important;
#     }
#     .stTabs [data-baseweb="tab-list"] {
#         gap: 10px;
#     }
#     .stTabs [data-baseweb="tab"] {
#         background-color: #f1f3f9;
#         border-radius: 4px 4px 0 0;
#         padding: 10px 20px;
#         border: none;
#     }
#     .stTabs [aria-selected="true"] {
#         background-color: #4e73df !important;
#         color: white !important;
#     }
#     .user-table {
#         width: 100%;
#         border-collapse: collapse;
#     }
#     .user-table th, .user-table td {
#         border: 1px solid #ddd;
#         padding: 8px;
#         text-align: left;
#     }
#     .user-table th {
#         background-color: #4e73df;
#         color: white;
#     }
#     .user-table tr:nth-child(even) {
#         background-color: #f2f2f2;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Function to get Lottie animations
# def load_lottieurl(url: str):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()

# # Lottie animation URLs
# face_scan_animation = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_rjgikbck.json")
# access_granted_animation = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_ukjcyybw.json")
# access_denied_animation = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_qpwbiyxf.json")
# add_user_animation = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_q77qtgsv.json")
# training_animation = load_lottieurl("https://assets3.lottiefiles.com/private_files/lf30_wqypnpu5.json")
# dashboard_animation = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_ysrn2iwp.json")

# # Initialize session state variables
# if 'passkey_attempts' not in st.session_state:
#     st.session_state.passkey_attempts = 0

# if 'access_granted' not in st.session_state:
#     st.session_state.access_granted = False

# if 'show_main_options' not in st.session_state:
#     st.session_state.show_main_options = False

# if 'dataset_info' not in st.session_state:
#     st.session_state.dataset_info = {'name': '', 'age': '', 'address': ''}

# if 'crud_operation_verified' not in st.session_state:
#     st.session_state.crud_operation_verified = False

# if 'operation_type' not in st.session_state:
#     st.session_state.operation_type = None

# if 'selected_user_id' not in st.session_state:
#     st.session_state.selected_user_id = None

# if 'user_to_edit' not in st.session_state:
#     st.session_state.user_to_edit = {'id': None, 'name': '', 'age': '', 'address': ''}

# # Add a new session state variable to store the authenticated user's name
# if 'authenticated_user' not in st.session_state:
#     st.session_state.authenticated_user = None

# # Add variable to track authentication method
# if 'auth_method' not in st.session_state:
#     st.session_state.auth_method = None

# # Create data directory if it doesn't exist
# data_dir = "data"
# if not os.path.exists(data_dir):
#     os.makedirs(data_dir)

# # Load Haar Cascade for face detection
# cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# faceCascade = cv2.CascadeClassifier(cascade_path)

# if faceCascade.empty():
#     st.error("Error: Haar Cascade XML file could not be loaded.")
#     st.stop()

# # Load LBPH Recognizer
# recognizer = cv2.face.LBPHFaceRecognizer_create()

# # Load classifier if it exists
# classifier_exists = False
# try:
#     recognizer.read("classifier.xml")
#     classifier_exists = True
# except Exception as e:
#     st.warning("Classifier not found. Train the model before detecting faces.")

# # Database connection function
# def get_db_connection():
#     try:
#         conn = mysql.connector.connect(
#             host="localhost",
#             user="root",
#             passwd="",
#             database="Authorized_users"
#         )
#         return conn
#     except Exception as e:
#         st.error(f"Database connection error: {e}")
#         return None

# # Initialize database if it doesn't exist
# def init_database():
#     try:
#         conn = mysql.connector.connect(
#             host="localhost",
#             user="root",
#             passwd=""
#         )
#         cursor = conn.cursor()
        
#         # Create database if it doesn't exist
#         cursor.execute("CREATE DATABASE IF NOT EXISTS Authorized_users")
#         cursor.execute("USE Authorized_users")
        
#         # Check if table exists
#         cursor.execute("SHOW TABLES LIKE 'my_table'")
#         table_exists = cursor.fetchone()
        
#         # Create table if it doesn't exist with explicit AUTO_INCREMENT definition
#         if not table_exists:
#             cursor.execute("""
#             CREATE TABLE my_table (
#                 id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
#                 name VARCHAR(255) NOT NULL,
#                 age VARCHAR(50),
#                 address TEXT
#             ) ENGINE=InnoDB AUTO_INCREMENT=1
#             """)
        
#         conn.commit()
#         return True
#     except Exception as e:
#         st.error(f"Database initialization error: {e}")
#         return False
#     finally:
#         if 'conn' in locals() and conn.is_connected():
#             cursor.close()
#             conn.close()

# # Email Alert Function
# def send_email_alert(person_name, confidence, img_path):
#     try:
#         email_sender = "samyak.1403@gmail.com"
#         email_password = "jpcngoqvqeapstnz"  # Consider using environment variables for this
#         email_receiver = "deepanshiajmera1304@gmail.com"

#         subject = "🔔 Face Detection Alert"
#         body = f"Person Detected: {person_name}\nConfidence Level: {confidence}%\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

#         msg = EmailMessage()
#         msg['From'] = email_sender
#         msg['To'] = email_receiver
#         msg['Subject'] = subject
#         msg.set_content(body)

#         with open(img_path, 'rb') as f:
#             img_data = f.read()
#             msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename='detected_face.jpg')

#         with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
#             server.login(email_sender, email_password)
#             server.send_message(msg)

#         st.success(f"Email alert sent for {person_name}")
#         return True
#     except Exception as e:
#         st.error(f"Email sending error: {e}")
#         return False

# # Detect Face Function
# def detect_face():
#     if not classifier_exists:
#         st.error("Face recognition model not trained yet. Please add users and train classifier first.")
#         return False
    
#     # Display scanning animation
#     animation_col, text_col = st.columns([1, 2])
#     with animation_col:
#         st_lottie(face_scan_animation, height=200, key="face_scan")
#     with text_col:
#         st.markdown("<h3>Face Recognition Authentication</h3>", unsafe_allow_html=True)
#         st.markdown("Please position your face in the camera frame for detection.")
    
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         st.error("Error: Cannot access the webcam. Check your camera permissions!")
#         return False

#     stframe = st.empty()
#     status_placeholder = st.empty()
#     progress_placeholder = st.empty()
#     start_time = datetime.now()
    
#     status_placeholder.info("Detecting face... Look at the camera")
    
#     # Track if we've found a valid face
#     face_detected = False
#     person_name = None
#     best_confidence = 0
    
#     # Create a progress bar for the timeout
#     progress_bar = progress_placeholder.progress(0)
#     timeout_seconds = 20

#     while (datetime.now() - start_time).seconds < timeout_seconds:
#         # Update progress bar
#         elapsed = (datetime.now() - start_time).seconds
#         progress_bar.progress(elapsed / timeout_seconds)
        
#         ret, img = cap.read()
#         if not ret:
#             status_placeholder.error("Failed to read frame from webcam.")
#             break

#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces = faceCascade.detectMultiScale(gray, 1.1, 5)

#         for (x, y, w, h) in faces:
#             cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
#             face_roi = gray[y:y+h, x:x+w]
            
#             try:
#                 id, pred = recognizer.predict(face_roi)
#                 confidence = int(100 * (1 - pred / 300))
                
#                 conn = get_db_connection()
#                 if not conn:
#                     status_placeholder.error("Could not connect to database")
#                     continue
                    
#                 cursor = conn.cursor()
#                 cursor.execute(f"SELECT name FROM my_table WHERE id={id}")
#                 user_data = cursor.fetchone()
#                 cursor.close()
#                 conn.close()

#                 if user_data:
#                     current_name = user_data[0]
#                     cv2.putText(img, f"{current_name} ({confidence}%)", (x, y-10), 
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    
#                     # Track the best confidence we've seen
#                     if confidence > best_confidence:
#                         best_confidence = confidence
#                         person_name = current_name
                        
#                     # Only set face_detected if we exceed the threshold
#                     if confidence > 70:
#                         face_detected = True
#                 else:
#                     cv2.putText(img, f"Unknown ({confidence}%)", (x, y-10), 
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    
#                     # Save unknown face and send alert after a few seconds
#                     if (datetime.now() - start_time).seconds > 5:
#                         img_path = os.path.join(data_dir, "unknown_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".jpg")
#                         cv2.imwrite(img_path, img)
#                         send_email_alert("Unknown", confidence, img_path)
#                         status_placeholder.error("Unauthorized Person Detected. Alert Sent.")

#             except Exception as e:
#                 status_placeholder.error(f"Recognition Error: {e}")

#         # Convert color for display
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         stframe.image(img_rgb, caption="Face Detection", use_column_width=True)
        
#         # Check if we've found a valid face with high confidence
#         if face_detected:
#             # Wait a bit longer to ensure we have a stable reading
#             if (datetime.now() - start_time).seconds > 3:
#                 break

#     cap.release()
    
#     # Clear the components used during scanning
#     progress_placeholder.empty()
#     stframe.empty()
    
#     # Only grant access if a valid face was detected
#     if face_detected and best_confidence > 70:
#         st.session_state.access_granted = True
#         # Store the authenticated user's name
#         st.session_state.authenticated_user = person_name
#         # Set authentication method to face
#         st.session_state.auth_method = "face"
#         # Auto-verify CRUD operations for face authentication
#         st.session_state.crud_operation_verified = True
        
#         # Show success animation
#         col1, col2 = st.columns([1, 2])
#         with col1:
#             st_lottie(access_granted_animation, height=200, key="access_granted")
#         with col2:
#             st.markdown(f"<h2 class='success-text'>Access Granted!</h2>", unsafe_allow_html=True)
#             st.markdown(f"<h3>Welcome, {person_name}!</h3>", unsafe_allow_html=True)
#             st.markdown(f"<p>Identified with {best_confidence}% confidence</p>", unsafe_allow_html=True)
        
#         # Add a small delay for better user experience
#         time.sleep(2)
#         st.experimental_rerun()
#         return True
#     else:
#         # Show failure animation
#         col1, col2 = st.columns([1, 2])
#         with col1:
#             st_lottie(access_denied_animation, height=200, key="access_denied")
#         with col2:
#             st.markdown("<h2 class='error-text'>Access Denied</h2>", unsafe_allow_html=True)
#             if best_confidence > 0:
#                 st.markdown(f"<p>Face detected but confidence too low ({best_confidence}%).</p>", unsafe_allow_html=True)
#             else:
#                 st.markdown("<p>No recognized face detected.</p>", unsafe_allow_html=True)
        
#         time.sleep(2)
#         return False
        
# # Dataset Generation Function
# def generate_dataset(user_id):
#     name = st.session_state.dataset_info['name']
    
#     # Create directory for user images
#     person_dir = os.path.join(data_dir, f"user_{user_id}")
#     if not os.path.exists(person_dir):
#         os.makedirs(person_dir)
    
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         st.error("Error: Cannot access the webcam. Check your camera permissions!")
#         return False
        
#     counter = 0
#     total_samples = 30
#     progress_bar = st.progress(0)
#     status_text = st.empty()
#     frame_display = st.empty()
    
#     status_text.info(f"Capturing face samples for {name}. Please look at the camera and move your head slightly.")
    
#     while counter < total_samples:
#         ret, img = cap.read()
#         if not ret:
#             st.error("Error capturing image from webcam.")
#             break
            
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces = faceCascade.detectMultiScale(gray, 1.1, 5)
        
#         if len(faces) > 0:  # Only save if a face is detected
#             for (x, y, w, h) in faces:
#                 cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
#                 # Save only the face region
#                 face_img = gray[y:y+h, x:x+w]
#                 img_path = os.path.join(person_dir, f"{counter}.jpg")
#                 cv2.imwrite(img_path, face_img)
                
#                 counter += 1
#                 progress_bar.progress(counter / total_samples)
#                 status_text.info(f"Capturing sample {counter}/{total_samples}")
                
#                 if counter >= total_samples:
#                     break
                
#         # Display the frame
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         frame_display.image(img_rgb, caption="Capturing face data", use_column_width=True)
            
#     cap.release()
    
#     if counter >= total_samples:
#         status_text.success(f"Dataset generated successfully with {counter} samples")
#         return True
#     else:
#         status_text.error(f"Only captured {counter}/{total_samples} samples. Try again.")
#         return False

# # Train Classifier Function
# def train_classifier():
#     global classifier_exists  # Properly declare global variable
    
#     status = st.empty()
#     status.info("Training classifier...")
    
#     progress_bar = st.progress(0)
    
#     face_samples = []
#     face_ids = []
    
#     # Connect to database to get user IDs
#     conn = get_db_connection()
#     if not conn:
#         status.error("Could not connect to database for training")
#         return False
        
#     cursor = conn.cursor()
#     cursor.execute("SELECT id, name FROM my_table")
#     db_users = cursor.fetchall()
#     cursor.close()
#     conn.close()
    
#     # Exit if no users
#     if not db_users:
#         status.error("No users found in database. Add users first.")
#         return False
    
#     # Load all face samples
#     for i, (user_id, name) in enumerate(db_users):
#         user_dir = os.path.join(data_dir, f"user_{user_id}")
#         if not os.path.exists(user_dir):
#             status.warning(f"No data directory found for {name} (ID: {user_id})")
#             continue
            
#         status.info(f"Loading training data for {name}...")
#         progress_bar.progress((i / len(db_users)) * 0.5)  # First half of progress is loading data
        
#         for img_file in os.listdir(user_dir):
#             if not img_file.endswith('.jpg'):
#                 continue
                
#             img_path = os.path.join(user_dir, img_file)
#             face_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
#             if face_img is not None:
#                 face_samples.append(face_img)
#                 face_ids.append(user_id)
    
#     if not face_samples:
#         status.error("No face samples found. Generate datasets first.")
#         return False
    
#     # Train the recognizer
#     status.info(f"Training with {len(face_samples)} images...")
    
#     # Simulate progress during training
#     for i in range(50, 100):
#         progress_bar.progress(i / 100)
#         time.sleep(0.05)  # Add a small delay to show progress
    
#     recognizer.train(face_samples, np.array(face_ids))
    
#     # Save trained model
#     recognizer.write("classifier.xml")
#     classifier_exists = True  # Set after successful training
#     progress_bar.progress(1.0)
#     status.success("Classifier trained successfully!")
    
#     # Display success message with animation
#     st.empty()
#     col1, col2 = st.columns([1, 2])
#     with col1:
#         st_lottie(training_animation, height=200, key="train_success")
#     with col2:
#         st.markdown("<h3 class='success-text'>Training Complete!</h3>", unsafe_allow_html=True)
#         st.markdown("<p>The face recognition system has been successfully trained with the provided datasets. The system is now ready to identify registered users.</p>", unsafe_allow_html=True)
    
#     return True

# # Add user to database
# def add_user_to_db():
#     name = st.session_state.dataset_info['name']
#     age = st.session_state.dataset_info['age']
#     address = st.session_state.dataset_info['address']
    
#     if not name or not age or not address:
#         st.error("Please provide complete details before submitting.")
#         return None
    
#     conn = get_db_connection()
#     if not conn:
#         return None
        
#     try:
#         cursor = conn.cursor()
#         # Explicitly specify columns to insert into
#         query = "INSERT INTO my_table (name, age, address) VALUES (%s, %s, %s)"
#         values = (name, age, address)
#         cursor.execute(query, values)
#         conn.commit()
        
#         # Get the ID of the inserted user
#         user_id = cursor.lastrowid
#         st.success(f"User {name} added to database with ID: {user_id}")
#         return user_id
#     except Exception as e:
#         st.error(f"Error adding user to database: {e}")
#         # If the error is about structure, try to fix the table
#         if "1364" in str(e):
#             st.warning("Attempting to fix database structure...")
#             try:
#                 # Drop and recreate the table with proper AUTO_INCREMENT
#                 cursor.execute("DROP TABLE IF EXISTS my_table")
#                 cursor.execute("""
#                 CREATE TABLE my_table (
#                     id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
#                     name VARCHAR(255) NOT NULL,
#                     age VARCHAR(50),
#                     address TEXT
#                 ) ENGINE=InnoDB AUTO_INCREMENT=1
#                 """)
#                 conn.commit()
                
#                 # Try inserting again
#                 query = "INSERT INTO my_table (name, age, address) VALUES (%s, %s, %s)"
#                 values = (name, age, address)
#                 cursor.execute(query, values)
#                 conn.commit()
                
#                 user_id = cursor.lastrowid
#                 st.success(f"Database fixed! User {name} added with ID: {user_id}")
#                 return user_id
#             except Exception as e2:
#                 st.error(f"Failed to fix database: {e2}")
#                 return None
#         return None
#     finally:
#         if 'cursor' in locals():
#             cursor.close()
#         conn.close()

# # Update user in database
# def update_user_in_db(user_id, name, age, address):
#     conn = get_db_connection()
#     if not conn:
#         return False
        
#     try:
#         cursor = conn.cursor()
#         query = "UPDATE my_table SET name = %s, age = %s, address = %s WHERE id = %s"
#         values = (name, age, address, user_id)
#         cursor.execute(query, values)
#         conn.commit()
        
#         if cursor.rowcount > 0:
#             st.success(f"User {name} (ID: {user_id}) updated successfully")
#             return True
#         else:
#             st.warning(f"No changes made to user {name} (ID: {user_id})")
#             return False
#     except Exception as e:
#         st.error(f"Error updating user: {e}")
#         return False
#     finally:
#         if 'cursor' in locals():
#             cursor.close()
#         conn.close()

# # Delete user from database and remove their images
# def delete_user_from_db(user_id):
#     # First get user info for confirmation
#     conn = get_db_connection()
#     if not conn:
#         return False
        
#     try:
#         cursor = conn.cursor()
        
#         # Get user name for confirmation
#         cursor.execute("SELECT name FROM my_table WHERE id = %s", (user_id,))
#         user = cursor.fetchone()
        
#         if not user:
#             st.error(f"User with ID {user_id} not found")
#             return False
            
#         user_name = user[0]
        
#         # Delete user from database
#         cursor.execute("DELETE FROM my_table WHERE id = %s", (user_id,))
#         conn.commit()
        
#         # Delete user's image directory
#         user_dir = os.path.join(data_dir, f"user_{user_id}")
#         if os.path.exists(user_dir):
#             shutil.rmtree(user_dir)
            
#         st.success(f"User {user_name} (ID: {user_id}) deleted successfully along with all face data")
#         return True
#     except Exception as e:
#         st.error(f"Error deleting user: {e}")
#         return False
#     finally:
#         if 'cursor' in locals():
#             cursor.close()
#         conn.close()

# # Get user by ID
# def get_user_by_id(user_id):
#     conn = get_db_connection()
#     if not conn:
#         return None
        
#     try:
#         cursor = conn.cursor(dictionary=True)
#         cursor.execute("SELECT id, name, age, address FROM my_table WHERE id = %s", (user_id,))
#         user = cursor.fetchone()
#         return user
#     except Exception as e:
#         st.error(f"Error fetching user: {e}")
#         return None
#     finally:
#         if 'cursor' in locals():
#             cursor.close()
#         conn.close()

# # Verify passkey for CRUD operations
# def verify_passkey_for_crud():
#     st.markdown("<div class='card'>", unsafe_allow_html=True)
#     st.markdown("<h3 class='sub-header'>Authentication Required</h3>", unsafe_allow_html=True)
#     st.warning("Please enter the passkey to access admin functions")
    
#     passkey = st.text_input("Enter Admin Passkey", type="password", key="crud_passkey")
    
#     col1, col2 = st.columns([1, 3])
#     with col1:
#         if st.button("Verify Passkey", key="verify_crud_btn"):
#             if passkey == "securepass123":
#                 st.session_state.crud_operation_verified = True
#                 st.success("Passkey verified. You can now perform any operation.")
#                 st.experimental_rerun()
#             else:
#                 st.error("Incorrect passkey. Access denied.")
#     st.markdown("</div>", unsafe_allow_html=True)

# # Get a card with user statistics
# def get_user_stats():
#     conn = get_db_connection()
#     if not conn:
#         return None
        
#     try:
#         cursor = conn.cursor()
#         cursor.execute("SELECT COUNT(*) FROM my_table")
#         total_users = cursor.fetchone()[0]
        
#         st.markdown("<div class='card'>", unsafe_allow_html=True)
#         st.markdown("<h3>System Statistics</h3>", unsafe_allow_html=True)
        
#         col1, col2 = st.columns(2)
#         with col1:
#             st.metric("Total Registered Users", total_users)
#         with col2:
#             st.metric("Face Model Status", "Trained" if classifier_exists else "Not Trained")
        
#         st.markdown("</div>", unsafe_allow_html=True)
        
#     except Exception as e:
#         st.error(f"Error getting statistics: {e}")
#     finally:
#         if 'cursor' in locals():
#             cursor.close()
#         conn.close()

# # Initialize database on app start
# init_database()

# # Main App
# st.markdown("<h1 class='main-header'>🔒 Advanced Face Recognition System</h1>", unsafe_allow_html=True)

# if not st.session_state.access_granted:
#     # Create two columns for a more organized layout
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         st.markdown("<div class='card'>", unsafe_allow_html=True)
#         st.markdown("<h2 class='sub-header'>Authentication Portal</h2>", unsafe_allow_html=True)
#         option = st.radio("Choose Authentication Method", ["Face Recognition", "Passkey"])

#         if option == "Face Recognition":
#             st.info("This method will use your webcam to verify your identity.")
#             if st.button("Start Face Recognition", key="start_face_recog"):
#                 detect_face()

#         elif option == "Passkey":
#             st.info("Enter the system passkey to gain access.")
#             passkey = st.text_input("Enter Passkey", type="password")

#             if st.button("Submit Passkey", key="submit_passkey"):
#                 if passkey == "securepass123":
#                     st.session_state.access_granted = True
#                     # Set authenticated user as Admin for passkey login
#                     st.session_state.authenticated_user = "Admin"
#                     # Set authentication method to passkey
#                     st.session_state.auth_method = "passkey"
#                     # Don't automatically verify CRUD operations for passkey auth
#                     st.session_state.crud_operation_verified = False
                    
#                     # Show success animation
#                     st_lottie(access_granted_animation, height=150, key="passkey_success")
#                     st.success("Access Granted via Passkey")
#                     time.sleep(1)
#                     st.experimental_rerun()
#                 else:
#                     st.session_state.passkey_attempts += 1
#                     st.error(f"Incorrect Passkey. Access Denied. ({st.session_state.passkey_attempts}/3 attempts)")

#                 if st.session_state.passkey_attempts >= 3:
#                     st.error("Too many failed attempts. Switching to face recognition.")
#                     option = "Face Recognition"
#                     st.experimental_rerun()
#         st.markdown("</div>", unsafe_allow_html=True)
    
#     with col2:
#         st.markdown("<div class='card'>", unsafe_allow_html=True)
#         st.markdown("<div class='card'>", unsafe_allow_html=True)
#         st.markdown("<h2 class='sub-header'>System Information</h2>", unsafe_allow_html=True)
        
#         # Add Lottie animation for dashboard
#         st_lottie(dashboard_animation, height=200, key="dashboard_anim")
        
#         st.markdown("""
#         <p>This advanced face recognition system provides:</p>
#         <ul>
#             <li>🔒 Secure biometric authentication</li>
#             <li>📊 User management dashboard</li>
#             <li>📨 Email alerts for unauthorized access</li>
#             <li>🔄 Easy user registration and training</li>
#         </ul>
#         <p>Choose an authentication method to continue.</p>
#         """, unsafe_allow_html=True)
#         st.markdown("</div>", unsafe_allow_html=True)

# else:
#     # Show user dashboard after successful authentication
#     # Sidebar for navigation
#     with st.sidebar:
#         st.markdown(f"<h3>Welcome, {st.session_state.authenticated_user}!</h3>", unsafe_allow_html=True)
#         st.markdown(f"<p>Authentication method: {st.session_state.auth_method.title()}</p>", unsafe_allow_html=True)
        
#         # Create a nice sidebar menu with icons
#         st.markdown("""
#         <style>
#             .sidebar-option {
#                 padding: 10px;
#                 border-radius: 5px;
#                 margin-bottom: 10px;
#                 cursor: pointer;
#                 transition: all 0.3s;
#             }
#             .sidebar-option:hover {
#                 background-color: rgba(78, 115, 223, 0.1);
#             }
#             .sidebar-icon {
#                 margin-right: 10px;
#                 font-size: 1.2rem;
#             }
#         </style>
#         """, unsafe_allow_html=True)
        
#         selected = st.radio(
#             "Navigation",
#             ["Dashboard", "User Management", "Face Detection", "System Settings"],
#             format_func=lambda x: f"{'📊' if x=='Dashboard' else '👥' if x=='User Management' else '🔍' if x=='Face Detection' else '⚙️'} {x}"
#         )
        
#         # Logout button
#         if st.button("Logout", key="logout_btn", help="Click to log out and return to login screen"):
#             st.session_state.access_granted = False
#             st.session_state.authenticated_user = None
#             st.session_state.auth_method = None
#             st.session_state.crud_operation_verified = False
#             st.experimental_rerun()
    
#     # Main content area based on sidebar selection
#     if selected == "Dashboard":
#         st.markdown("<h2 class='sub-header'>📊 System Dashboard</h2>", unsafe_allow_html=True)
        
#         # Dashboard metrics in cards with animations
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.markdown("<div class='card'>", unsafe_allow_html=True)
#             st.metric("System Status", "Online", delta="+99.8% Uptime")
#             st.metric("Last Login", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
#             st.markdown("</div>", unsafe_allow_html=True)
            
#         with col2:
#             get_user_stats()
        
#         # Access logs
#         st.markdown("<div class='card'>", unsafe_allow_html=True)
#         st.markdown("<h3>Recent Access Logs</h3>", unsafe_allow_html=True)
        
#         # Generate dummy access logs for demonstration
#         log_data = {
#             "Timestamp": [
#                 (datetime.now() - timedelta(minutes=i*15)).strftime("%Y-%m-%d %H:%M:%S")
#                 for i in range(5)
#             ],
#             "User": ["Admin", "John Doe", "Jane Smith", "Admin", "Unknown"],
#             "Access Type": ["Passkey", "Face Recognition", "Face Recognition", "Passkey", "Failed Attempt"],
#             "Status": ["Successful", "Successful", "Successful", "Successful", "Denied"]
#         }
        
#         # Convert to DataFrame for better display
#         import pandas as pd
#         log_df = pd.DataFrame(log_data)
        
#         # Style the dataframe
#         def color_status(val):
#             color = 'green' if val == 'Successful' else 'red' if val == 'Denied' else 'black'
#             return f'color: {color}; font-weight: bold'
        
#         styled_logs = log_df.style.applymap(color_status, subset=['Status'])
#         st.dataframe(styled_logs, use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
        
#         # System health chart
#         st.markdown("<div class='card'>", unsafe_allow_html=True)
#         st.markdown("<h3>System Performance</h3>", unsafe_allow_html=True)
        
#         # Create a simple chart for demonstration
#         # Create a simple chart for demonstration
#         chart_data = pd.DataFrame({
#             'Date': pd.date_range(start=datetime.datetime.now() - datetime.timedelta(days=6), periods=7, freq='D'),  # Added comma here
#             'Recognition Speed (ms)': [120, 118, 125, 110, 105, 102, 100],
#             'Accuracy (%)': [95.2, 95.8, 94.5, 96.2, 97.1, 97.8, 98.5]
#         })
        
#         chart_data = chart_data.melt('Date', var_name='Metric', value_name='Value')
        
#         import altair as alt
#         chart = alt.Chart(chart_data).mark_line(point=True).encode(
#             x='Date:T',
#             y='Value:Q',
#             color='Metric:N',
#             tooltip=['Date', 'Metric', 'Value']
#         ).interactive()
        
#         st.altair_chart(chart, use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
            
#     elif selected == "User Management":
#         st.markdown("<h2 class='sub-header'>👥 User Management</h2>", unsafe_allow_html=True)
        
#         # Check if we need to verify for CRUD operations based on auth method
#         if st.session_state.auth_method == "passkey" and not st.session_state.crud_operation_verified:
#             # Need to verify with passkey first
#             verify_passkey_for_crud()
#         else:
#             # No verification needed for face recognition users or already verified passkey users
#             tabs = st.tabs(["View Users", "Add User", "Edit User", "Delete User"])
            
#             with tabs[0]:
#                 st.markdown("<h3>Registered Users</h3>", unsafe_allow_html=True)
                
#                 # Fetch all users
#                 conn = get_db_connection()
#                 if conn:
#                     try:
#                         cursor = conn.cursor()
#                         cursor.execute("SELECT id, name, age, address FROM my_table")
#                         users = cursor.fetchall()
                        
#                         if users:
#                             st.markdown("<table class='user-table'>", unsafe_allow_html=True)
#                             st.markdown("<tr><th>ID</th><th>Name</th><th>Age</th><th>Address</th><th>Actions</th></tr>", unsafe_allow_html=True)
                            
#                             for user in users:
#                                 user_id, name, age, address = user
#                                 st.markdown(f"""
#                                 <tr>
#                                     <td>{user_id}</td>
#                                     <td>{name}</td>
#                                     <td>{age}</td>
#                                     <td>{address}</td>
#                                     <td>
#                                         <button onclick="alert('Edit function would be here')">Edit</button>
#                                         <button onclick="alert('Delete function would be here')">Delete</button>
#                                     </td>
#                                 </tr>
#                                 """, unsafe_allow_html=True)
                            
#                             st.markdown("</table>", unsafe_allow_html=True)
#                         else:
#                             st.info("No users registered yet.")
#                     except Exception as e:
#                         st.error(f"Error fetching users: {e}")
#                     finally:
#                         cursor.close()
#                         conn.close()
            
#             with tabs[1]:
#                 st.markdown("<h3>Add New User</h3>", unsafe_allow_html=True)
                
#                 # Display add user animation
#                 st_lottie(add_user_animation, height=200, key="add_user_anim")
                
#                 st.markdown("<div class='card'>", unsafe_allow_html=True)
#                 st.text_input("Full Name", key="new_user_name", 
#                               on_change=lambda: setattr(st.session_state.dataset_info, 'name', st.session_state.new_user_name))
#                 st.text_input("Age", key="new_user_age", 
#                               on_change=lambda: setattr(st.session_state.dataset_info, 'age', st.session_state.new_user_age))
#                 st.text_area("Address", key="new_user_address", 
#                              on_change=lambda: setattr(st.session_state.dataset_info, 'address', st.session_state.new_user_address))
                
#                 # Update session state
#                 st.session_state.dataset_info['name'] = st.session_state.get("new_user_name", "")
#                 st.session_state.dataset_info['age'] = st.session_state.get("new_user_age", "")
#                 st.session_state.dataset_info['address'] = st.session_state.get("new_user_address", "")
                
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     if st.button("Add to Database", key="add_user_db_btn"):
#                         user_id = add_user_to_db()
#                         if user_id:
#                             st.session_state.operation_type = "add"
#                             st.session_state.selected_user_id = user_id
#                 with col2:
#                     if st.session_state.operation_type == "add" and st.session_state.selected_user_id:
#                         if st.button("Generate Face Dataset", key="gen_dataset_btn"):
#                             success = generate_dataset(st.session_state.selected_user_id)
#                             if success:
#                                 # Reset after successful dataset generation
#                                 st.session_state.operation_type = None
#                                 st.session_state.selected_user_id = None
                                
#                                 # Offer to train classifier
#                                 train_btn = st.button("Train Classifier with New Data", key="train_after_add")
#                                 if train_btn:
#                                     train_classifier()
#                 st.markdown("</div>", unsafe_allow_html=True)
            
#             with tabs[2]:
#                 st.markdown("<h3>Edit Existing User</h3>", unsafe_allow_html=True)
                
#                 # Fetch all users for selection
#                 conn = get_db_connection()
#                 user_options = []
                
#                 if conn:
#                     try:
#                         cursor = conn.cursor()
#                         cursor.execute("SELECT id, name FROM my_table")
#                         users = cursor.fetchall()
                        
#                         if users:
#                             user_options = [f"{user[0]} - {user[1]}" for user in users]
#                         else:
#                             st.info("No users to edit.")
#                     except Exception as e:
#                         st.error(f"Error fetching users: {e}")
#                     finally:
#                         cursor.close()
#                         conn.close()
                
#                 if user_options:
#                     selected_user = st.selectbox("Select User to Edit", user_options, key="edit_user_select")
                    
#                     if selected_user:
#                         # Extract user ID from selection
#                         user_id = int(selected_user.split(" - ")[0])
                        
#                         # Fetch user details
#                         user = get_user_by_id(user_id)
                        
#                         if user:
#                             st.markdown("<div class='card'>", unsafe_allow_html=True)
#                             st.text_input("Full Name", value=user['name'], key="edit_name")
#                             st.text_input("Age", value=user['age'], key="edit_age")
#                             st.text_area("Address", value=user['address'], key="edit_address")
                            
#                             if st.button("Update User Information", key="update_user_btn"):
#                                 success = update_user_in_db(
#                                     user_id,
#                                     st.session_state.edit_name,
#                                     st.session_state.edit_age,
#                                     st.session_state.edit_address
#                                 )
#                                 if success:
#                                     st.success(f"User information updated. Consider retraining the classifier.")
                                    
#                                     # Offer to retrain
#                                     if st.button("Retrain Classifier", key="retrain_after_edit"):
#                                         train_classifier()
#                             st.markdown("</div>", unsafe_allow_html=True)
            
#             with tabs[3]:
#                 st.markdown("<h3>Delete User</h3>", unsafe_allow_html=True)
                
#                 # Fetch all users for selection
#                 conn = get_db_connection()
#                 user_options = []
                
#                 if conn:
#                     try:
#                         cursor = conn.cursor()
#                         cursor.execute("SELECT id, name FROM my_table")
#                         users = cursor.fetchall()
                        
#                         if users:
#                             user_options = [f"{user[0]} - {user[1]}" for user in users]
#                         else:
#                             st.info("No users to delete.")
#                     except Exception as e:
#                         st.error(f"Error fetching users: {e}")
#                     finally:
#                         cursor.close()
#                         conn.close()
                
#                 if user_options:
#                     col1, col2 = st.columns([2, 1])
                    
#                     with col1:
#                         selected_user = st.selectbox("Select User to Delete", user_options, key="delete_user_select")
                    
#                     if selected_user:
#                         # Extract user ID from selection
#                         user_id = int(selected_user.split(" - ")[0])
#                         user_name = selected_user.split(" - ")[1]
                        
#                         st.warning(f"⚠️ Are you sure you want to delete {user_name}? This action cannot be undone.")
#                         st.info("This will remove the user from the database and delete all associated face images.")
                        
#                         # Two-step deletion process
#                         if st.button("Confirm Deletion", key="confirm_delete_btn"):
#                             success = delete_user_from_db(user_id)
#                             if success:
#                                 # Offer to retrain the classifier
#                                 if st.button("Retrain Classifier After User Deletion", key="retrain_after_delete"):
#                                     train_classifier()
    
#     elif selected == "Face Detection":
#         st.markdown("<h2 class='sub-header'>🔍 Face Detection Portal</h2>", unsafe_allow_html=True)
        
#         tabs = st.tabs(["Verify Identity", "Train Classifier"])
        
#         with tabs[0]:
#             st.markdown("<h3>Verify Identity</h3>", unsafe_allow_html=True)
#             st.markdown("<div class='card'>", unsafe_allow_html=True)
#             st.write("Use this feature to verify a person's identity using face recognition.")
            
#             if st.button("Start Face Verification", key="face_verify_btn"):
#                 detect_face()
#             st.markdown("</div>", unsafe_allow_html=True)
        
#         with tabs[1]:
#             st.markdown("<h3>Train Face Recognition Model</h3>", unsafe_allow_html=True)
#             st.markdown("<div class='card'>", unsafe_allow_html=True)
#             st.write("Train the face recognition model with all available user data.")
#             st.info("This process will improve recognition accuracy. Run this after adding or updating users.")
            
#             if st.button("Start Training", key="train_clf_btn"):
#                 train_classifier()
#             st.markdown("</div>", unsafe_allow_html=True)
    
#     elif selected == "System Settings":
#         st.markdown("<h2 class='sub-header'>⚙️ System Settings</h2>", unsafe_allow_html=True)
        
#         st.markdown("<div class='card'>", unsafe_allow_html=True)
#         st.markdown("<h3>Email Notification Settings</h3>", unsafe_allow_html=True)
        
#         email_to = st.text_input("Notification Email", value="deepanshiajmera1304@gmail.com")
#         st.markdown("</div>", unsafe_allow_html=True)
        
#         st.markdown("<div class='card'>", unsafe_allow_html=True)
#         st.markdown("<h3>Recognition Settings</h3>", unsafe_allow_html=True)
        
#         confidence_threshold = st.slider("Recognition Confidence Threshold (%)", 50, 95, 70)
#         camera_source = st.selectbox("Camera Source", ["Default (0)", "External Camera (1)"])
        
#         if st.button("Save Settings", key="save_settings_btn"):
#             st.success("Settings saved successfully!")
#         st.markdown("</div>", unsafe_allow_html=True)
        
#         # System maintenance options
#         st.markdown("<div class='card'>", unsafe_allow_html=True)
#         st.markdown("<h3>System Maintenance</h3>", unsafe_allow_html=True)
        
#         if st.button("Backup User Database", key="backup_db_btn"):
#             st.success("Database backup created successfully!")
#             # Display a fake download button
#             st.markdown(
#                 f"""
#                 <div style="margin-top: 10px;">
#                     <a href="#" download="user_database_backup_{datetime.now().strftime('%Y%m%d')}.sql" 
#                        style="text-decoration: none;">
#                         <button style="background-color: #4CAF50; color: white; padding: 10px 20px; 
#                                  border: none; border-radius: 5px; cursor: pointer;">
#                             Download Backup File
#                         </button>
#                     </a>
#                 </div>
#                 """, 
#                 unsafe_allow_html=True
#             )
        
#         if st.button("Clear Unknown Face Images", key="clear_unknown_btn"):
#             # Count files that start with "unknown_"
#             unknown_files = [f for f in os.listdir(data_dir) if f.startswith("unknown_")]
#             if unknown_files:
#                 for f in unknown_files:
#                     os.remove(os.path.join(data_dir, f))
#                 st.success(f"Removed {len(unknown_files)} unknown face images!")
#             else:
#                 st.info("No unknown face images to clear.")
#         st.markdown("</div>", unsafe_allow_html=True)

# # Footer with credits
# st.markdown("""
# <div style="text-align: center; margin-top: 30px; padding: 10px; border-top: 1px solid #eee;">
#     <p>© 2025 Secure Face Recognition System | Developed with ❤️ by AI Team</p>
# </div>
# """, unsafe_allow_html=True)

# # Add a nice floating help button
# st.markdown("""
# <div style="position: fixed; bottom: 20px; right: 20px; z-index: 1000;">
#     <button style="background-color: #4e73df; color: white; width: 50px; height: 50px; 
#                  border-radius: 50%; border: none; font-size: 24px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
#         ?
#     </button>
# </div>
# """, unsafe_allow_html=True)