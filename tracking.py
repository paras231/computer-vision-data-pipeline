# !pip install ultralytics

from ultralytics import YOLO

model = YOLO("yolo11n.pt")


results = model.track(source="people_walking.mp4", tracker="my_bytetrack.yaml", persist=True, save=True)


# print(f"boxex : {results[0].boxes.xyxy}")

unique_ids = set()

for r in results:
  if r.boxes.id is not None:
    unique_ids.update(r.boxes.id.tolist())

print(unique_ids)
print(f"total people with id : {len(unique_ids)}")