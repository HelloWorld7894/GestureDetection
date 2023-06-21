# GestureDetection


Projekt na Týdnu Vědy na Jaderce ohledně rozpoznávání gest z kamery pomocí předtrénované neuronové sítě na odhad pózy. Testováno v Dockeru na NVIDIA Jetson Nano.
Jako příklad jsme naprogramovali hru **kámen, nůžky, papír** s AI nepřítelem a detekcí gest z kamery.

Předtrénované sítě:
1. resnet18-hand
2. resnet18-body
3. densenet121

TODO:
- [ ] refactor and typo fix
- [ ] natrénovat neuronovou síť na detekci gest
- [ ] migrovat na pc (použití mediapipe backbone)
