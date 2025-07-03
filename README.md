# car-repaint-detection

### Problem statement:-
To identify whether a car part is repainted or not

### Approach:-
1. Identify the dominant colours in a car part.
2. Look for variation in the most dominant colour, using the knowledge of the model and build of the car.
3. If variation exceeds a certain threshold, then the car part must be repainted, considering there won't be much variation in a non-repainted car part.

### Identifying Dominant Colours:
We can use effective clustering methods (k-means) to filter out the most dominant colour in a car part that covers most of the car part's area. For example: In a polar white car, the most dominant colour that covers the entire car part is white and its different shades.
We would create a single cluster for similar colours: this way, we can identify the colour that covers most of the car part

![image](https://github.com/user-attachments/assets/d4fd5440-cb2c-474d-9b0c-3ad87dbba9cf)

### Looking for variations:
When we have our dominant colour, we can check for variations in it, 
For example, if we get the most dominant colour as white that covers most of the car part area, we can check for how many pixels of different white shades are being covered like snow, off white, peach, etc.
After differentiating between whites(based on pixels covered) the present data is, we can calculate a variance of whites from the original colour code. 

![image](https://github.com/user-attachments/assets/14908420-b39a-4476-acd8-57b7c722423c)

### Thresholding:
We can experiment on already present data to calculate for variations in colours in a repainted car part and a non repainted car part to calculate a certain threshold . 
Exceeding that threshold would lead to more colour variations, and hence we can conclude that the car part is repainted.

Considering there are 6 shades of similar colours present in the crop

![image](images\image.png)

![image](images\image-1.png)

Total pixels=pixels of dominant colour

![image](images\image-2.png)

![image](images\image-3.png)

The final single variance value will give us an overall measure of how much the colors deviate from pure white, accounting for how common each shade is in the image.

### Why variation?
Consider white (255,255,255)
Small deviations like 255, 255, 254 can be considered a shade of white. Generally, any color where all three channels (R, G, B) are very close to 255 would be perceived as a variation of white. It’s not necessary for the channels to exactly be 255 to be considered white, as slight reductions (in the 250-255 range) will still fall under the category of "white" in most cases.

### Limitations:  
1. High definition image with good pixel values is required for better results.
2. If the image quality is poor, it will affect the variance and hence the results would be affected.
3. If the repaint is done with effective colour matching, then it wouldn't be possible to determine if the part is painted on pixel level(possible solution can be to check for dent scratch history of car) 
4. Fails where the repaint is of totally different colour, for example black colour on white as clustering only groups similar colours
5. If image quality is varying(which may be possible) then variations would be affected. (If original colours of car are not retained in the image, then results would alter)
