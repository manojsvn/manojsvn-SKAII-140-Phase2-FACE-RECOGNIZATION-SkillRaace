import os
import face_recognition
import cv2
import numpy as np
import pickle

print("libraries are imported")
# Path to the dataset directory
dataset_dir = 'C:\\Users\\nikhi\\Desktop\\face_recognisation\\Original Images\\Original Images'

# Initialize lists to hold known face encodings and their names
known_face_encodings = []
known_face_names = []

print("For loop is running")
c=0
# Loop through each person in the dataset directory

for person_name in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person_name)
    print(str(c+1)+" "+ person_dir)
    # Loop through each image file for the current person
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        
        # Load the image
        image = face_recognition.load_image_file(image_path)
        
        # Get the face encodings for the face in the image
        face_encodings = face_recognition.face_encodings(image)
        
        # Assuming each image has exactly one face
        if len(face_encodings) > 0:
            face_encoding = face_encodings[0]
            
            # Add the face encoding and the name of the person to the lists
            known_face_encodings.append(face_encoding)
            known_face_names.append(person_name)
    c=c+1

print("For loop ended")
print("Saving the model")
# Save the trained model data using pickle
with open('trained_face_recognition_model.pkl', 'wb') as f:
    pickle.dump((known_face_encodings, known_face_names), f)

# Function to load the trained model
def load_trained_model(model_path):
    with open(model_path, 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)
    return known_face_encodings, known_face_names

# Function to recognize faces in a given image
def recognize_faces(image_path, model_path):
    # Load the trained model
    known_face_encodings, known_face_names = load_trained_model(model_path)
    
    # Load the image
    image = face_recognition.load_image_file(image_path)
    
    # Find all the faces and face encodings in the image
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    # Convert the image to BGR color (which OpenCV uses)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for any known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        
        name = "Unknown"
        
        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        # Draw a box around the face
        cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Draw a label with the name below the face
        cv2.rectangle(image_bgr, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image_bgr, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    # Display the resulting image
    cv2.imshow('Image', image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
# Save the model first (uncomment below line to save)
# recognize_faces('C:\\Users\\nikhi\\Desktop\\face_recognisation\\Faces\\Faces\\Akshay Kumar_1.jpg', 'trained_face_recognition_model.pkl')

print("Testing")
# Load the model and recognize faces
recognize_faces('C:\\Users\\nikhi\\Desktop\\face_recognisation\\Faces\\Faces\\Akshay Kumar_1.jpg', 'trained_face_recognition_model.pkl')
# recognize_faces('C:\\Users\\nikhi\\Desktop\\face_recognisation\\Faces\\Faces\\Alexandra Daddario_17.jpg','trained_face_recognition_models.pkl')
