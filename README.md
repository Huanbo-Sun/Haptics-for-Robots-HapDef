# Haptics-with-Strain-Gauge
This project aims at showing the principle design of [virtual sensing](https://en.wikipedia.org/wiki/Virtual_sensing) in robotic applications with haptic functionality using **sparse sensor configuration**.

It includes five major parts:
- Robot's limb design, manufacturing in mechanical aspect.
- Sensor choice, positioning, assembly, data acquision in mechatronic aspect.
- Data postprocessing in functionality aspect.
- Automatic data collection system in application aspect.
- System integration in robotic application aspect.

## Research agent
We adapt one open-source project ["Poppy Project-Humanoids"](https://www.poppy-project.org/en/) and make contributions to optimize the hardware, software, and web tools.

In this repository, we introduce haptic feedback in robot's limb, which realizes single-contact and **multi-contact** stimulation localization and quantifization functions.

<img src="Pics/Project_object.png" width="800" height="400" align="center">

## Robot's limb design, manufacturing in mechanical aspect
### Limb design in [Solidworks](https://www.solidworks.com/de)
- Keep kinematic parameters unchanged
- In Solidworks, [parametrize](http://help.solidworks.com/2017/english/solidworks/cworks/parameters_2.htm) thickness of the flexible sensing shell and the in-middle placed structure support.
### Limb design validation in [ANSYS: Workbench-Static Structure](https://www.ansys.com/products/structures)
- Robustness check, whether the strucutre can support the  whole weight of the robot during its dynamics motion.
- Flexiablity check, how flexible the outer-shell of the limb is agagist stimulation.
<img src="Pics/Limb_design_validation.png" width="350" height="400" align="center">
###


