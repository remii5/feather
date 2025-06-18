## Motivations

I am equally as interested in improving lives with programming as I am interested in documenting life with a camera. I primarily shoot with a film camera on **B&W** film stocks and have learned to develop and scan my own negatives. Traditional film scanners utilize technology known as [Digital ICE](https://en.wikipedia.org/wiki/Digital_ICE). Developed by Kodak in the 60/70s this technology effectively cleans **C41 (color)** film scans by using an infrared scan to detect objects, namely dust/scratches on film. My emphasis on what I shoot and what the technology scans best is important because when it comes to B&W, the makeup of these film stocks have silver halide particles. These infrared scan does not recognize the particles and thus makes it difficult to remove the objects.

While I was not willing to sacrifice my love for B&W for the increased time it takes to ensure there are no unwanted objects on the scans, I did wish there was a way to decrease the time it would take to remove them. There are already existing applications that allow you to remove unwanted things from your image via software (e.g. Photoshop, Lightroom), but they still require human interaction. So in an effort to do this process WITHOUT manual intervention feather was born.

## Dataset

The dataset the model is trained on consists of 50+ images. These images derive from [Kodak PhotoCD Dataset](https://r0k.us/graphics/kodak/), film forums, and scanned b&w images of my own. Some images were synthetically given defects to increase the img count and every image was labeled using the LabelMe software.

## Results

Here is an example 256x256 result from an input/output image:
![256x256 input image and detected defects mask](https://private-user-images.githubusercontent.com/90795841/455223905-db1ace41-12ac-43d7-ac48-f2340f148b3f.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDk5NTQwNzIsIm5iZiI6MTc0OTk1Mzc3MiwicGF0aCI6Ii85MDc5NTg0MS80NTUyMjM5MDUtZGIxYWNlNDEtMTJhYy00M2Q3LWFjNDgtZjIzNDBmMTQ4YjNmLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA2MTUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNjE1VDAyMTYxMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTA4MTI5YmE3NjA0MzdjY2QxMmM4ZWU5MGIxYjNlYWQ0ZjE1OTBhM2RhMTZkMDBmMjRjYzM1MmMyYjFmM2I5MDgmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.6zns17tYdvVtk_Bs5BXsiag-G4rfGbiEUnxwFL5tUfU)
![256x256 reconstructed image](https://private-user-images.githubusercontent.com/90795841/455223906-7bce6488-e891-40a0-aff9-79d481913a06.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDk5NTQwNzIsIm5iZiI6MTc0OTk1Mzc3MiwicGF0aCI6Ii85MDc5NTg0MS80NTUyMjM5MDYtN2JjZTY0ODgtZTg5MS00MGEwLWFmZjktNzlkNDgxOTEzYTA2LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA2MTUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNjE1VDAyMTYxMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTA3ZmM3ZjlhOWI4NTJiYWE1MjZhOGY0MDFkMzg5MmE0Mjc4ZDk1M2Q2ODE1ZTE5ODZiM2E3MDk5YWY0N2NkNGEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.DRyOcS8rEbgBkzTOOS_Q36lnKE9JWilAj8F-5_TxE1U)

## Main Struggles & Takeaways

The world of Machine Learning and Digital Image Processing is massive, and is flooded with papers, videos, and in-depth resources. These resources, however, come with concepts and lingo I was never familiar with. This took up a majority of my time during this project period. My solution was to tackle the ideas I struggled with as I came across them in my research.

In regards to my programming, lots of time was spent debugging my reconstruction algorithm. I did not realize for a long time that it was only reconstructing a portion of my input image because of the size I had my algorithm analyse for defects.

I look forward to returning to this project or a similar project in the future. Finding ways to possibly optimize something like this would be interesting. This particular project would definitely see success with a larger dataset and optimizations in processing larger image file sizes.

## Requirements

If you are looking to try out this project the following folders and folder structure will be required:

- data/imgs
- data/masks/SegmentationClass
- input/
- output/

## Acknowledgements:

- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [Dust/Scratch Removal Paper](https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-17/issue-1/013010/Comprehensive-solutions-for-automatic-removal-of-dust-and-scratches-from/10.1117/1.2899845.short)
- [GDL Function Paper](https://arxiv.org/abs/1707.03237v3)
