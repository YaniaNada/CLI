import argparse
import joblib
import exercise_4.utils as utils
import cv2

def_image_path = 'exercise_4/sample_images/test_image.png'
target_names = joblib.load('exercise_4/target_names.pkl')

def main():
    parser = argparse.ArgumentParser(description = 'Enter the model type (RF for Random Forest, SVC for Support Vector Machine)')
    parser.add_argument('--model', default= 'SVC', help='The model type to run')
    parser.add_argument('--image', default=def_image_path, help='Path to the image file')
    args = parser.parse_args()
    
    if args.model == 'RF':
        model = joblib.load(f'exercise_4/best_rf_model.pkl')

    elif args.model == 'SVC':
        model = joblib.load(f'exercise_4/best_svc_model.pkl')

    input_image = cv2.imread(args.image)
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (62, 47))
    final_image = resized_image.reshape(1, -1)  
    pred_class_index = model.predict(final_image)[0]
    predicted_name = f"model selected: {args.model}, predicted labels: {pred_class_index}, {target_names[pred_class_index]}, (image file path: {args.image})"
    utils.write_fn(predicted_name, 'exercise_4/output.txt ')

if __name__ == '__main__':
    main()


