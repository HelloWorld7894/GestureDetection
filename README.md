# GestureDetection


Projekt na Týdnu Vědy na Jaderce ohledně rozpoznávání gest z kamery pomocí předtrénované neuronové sítě na odhad pózy. Testováno v Dockeru na NVIDIA Jetson Nano.
Jako příklad jsme naprogramovali hru **kámen, nůžky, papír** s AI nepřítelem a detekcí gest z kamery.

Předtrénované sítě:
1. resnet18-hand
2. resnet18-body
3. densenet121

## Schéma bodů na ruce
![hand_landmarks](https://github.com/HelloWorld7894/GestureDetection/blob/main/docs/hand_labels.png)

## Demo:

https://github.com/HelloWorld7894/GestureDetection/assets/59885570/89d24546-08a4-4a1d-8a52-ba32e7dc3506

## Odkazy:

odkaz na prezentaci &#8594; [https://github.com/HelloWorld7894/GestureDetection/blob/main/docs/presentation.pdf](https://github.com/HelloWorld7894/GestureDetection/blob/main/docs/presentation.pdf)\
odkaz na příspěvek do sborníku &#8594; [https://github.com/HelloWorld7894/GestureDetection/blob/main/docs/paper.pdf](https://github.com/HelloWorld7894/GestureDetection/blob/main/docs/paper.pdf)

---

## TODO:
- [x] docs update
- [ ] refactor and typo fix
- [ ] natrénovat neuronovou síť na detekci gest
- [ ] migrovat na pc (použití mediapipe backbone)
