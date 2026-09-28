def chatbot_response(text=None, image=None):

    # IMAGE PATH
    if image is not None:
        from image_predict import predict_image
        disease = predict_image(image)
        return {
            "mode": "image",
            "prediction": disease,
            "message": f" Based on the image, the disease is **{disease}**."
        }

    # TEXT PATH
    if text:
        from text_predict import predict_text
        disease = predict_text(text)
        return {
            "mode": "text",
            "prediction": disease,
            "message": f" Based on the symptoms, the disease is **{disease}**."
        }

    return {
        "message": "Please upload an image or enter plant symptoms."
    }