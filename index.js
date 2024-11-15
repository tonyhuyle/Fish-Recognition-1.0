const classNames = [
    'Atlantic Crevalle Jack', 
    'Atlantic Croaker',  
    'Black Drum',  
    'Gafftopsail Catfish',  
    'Ladyfish',  
    'Red Drum', 
    'Sheepshead',
    'Southern Flounder',  
    'Southern Kingfish', 
    'Spotted Seatrout',   
];

document.getElementById('uploadForm').onsubmit = async function(e) {
    e.preventDefault();
    const formData = new FormData(this);
    const response = await fetch('/predict', {
        method: 'POST',
        body: formData,
    });
    const result = await response.json();
    document.getElementById('result').innerText = 'Predicted Class: ' + result.class;
};

// Load the trained model
let model;
async function loadModel() {
    model = await tf.loadLayersModel('tfjs_files/model.json'); // Load the model
    console.log('Model loaded');
}

// Display the image preview when a file is selected
document.querySelector('input[type="file"]').onchange = function(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const imgPreview = document.getElementById('imagePreview');
            imgPreview.src = e.target.result;  // Set the image source to the uploaded file
            imgPreview.style.display = 'block'; // Show the image

            // Show the "Predict Species" button after image upload
            document.getElementById('predictButton').style.display = 'block';
        };
        reader.readAsDataURL(file); // Read the uploaded file as a data URL
    }
};

// Trigger prediction when "Predict Species" button is pressed
document.getElementById('predictButton').onclick = async function() {
    // Wait for the model to load if it's not loaded yet
    if (!model) {
        await loadModel();
    }

    // Get the image from the preview
    const img = document.getElementById('imagePreview');
    
    // Process the image and make a prediction
    const tensor = tf.browser.fromPixels(img)  // Convert image to tensor
        .resizeNearestNeighbor([224, 224])  // Resize the image to the model's input size
        .toFloat()
        .div(tf.scalar(255))  // Normalize pixel values to [0, 1]
        .expandDims();

    const prediction = model.predict(tensor);  // Get the prediction
    
    // Get the class probabilities
    const probabilities = prediction.dataSync(); // An array of probabilities for each class

    // Find the class index with the highest probability
    const classIndex = prediction.argMax(-1).dataSync()[0];  // Get the index of the predicted class

    // Get the class name using the index
    const predictedClassName = classNames[classIndex];

    // Display the predicted class name and probabilities
    const resultElement = document.getElementById('result');
    if (resultElement) {
        resultElement.innerText = `Predicted Class: ${predictedClassName}\n`;

        // Add probabilities for each class
        classNames.forEach((className, index) => {
            const percentage = (probabilities[index] * 100).toFixed(2);  // Convert to percentage
            resultElement.innerText += `${className}: ${percentage}%\n`;
        });
    } else {
        console.error("The element with id 'result' was not found.");
    }
};
