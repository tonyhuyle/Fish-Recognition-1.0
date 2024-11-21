## Fish Recognition Model

![image](https://github.com/user-attachments/assets/6daed115-3002-4104-8009-f8bbe7b0b497)

This model is trained to identify 10 different fish species commonly found in Galveston 

Specifically:
- Red Drum
- Spotted Seatrout
- Black Drum
- Spotted Seatrout
- Southern Flounder
- Southern Kingfish (Whiting)
- Atlantic Croaker
- Gafftopsail Catfish
- Sheepshead
- Ladyfish
- Cervalle Jack

# Prerequisites

 To use new data to train model:

```
pip install -r requirements.txt
```
Change data_dir to desired dataset
```
data_dir = 'dataset'
```

# Model Training
Begin model training by running (replace with respective model name) in terminal:

```
python model1.py
```
Finish by converting .h5 to .json

# Usage
In order to use the model, open the HTML file using live server and insert image of choice

![image](https://github.com/user-attachments/assets/f02d76d3-d1b0-4979-825b-d9e9e91c9a37)

Upon doing so, hit the predict button and allow the model to load before getting the results which show the top 3 possibilities.

![image](https://github.com/user-attachments/assets/92c738e7-db69-44bb-aab7-e96e1f92e578)



