# Haptics-with-Strain-Gauge
This project shows the principle design of [virtual sensing](https://en.wikipedia.org/wiki/Virtual_sensing) in robotic system with haptic feedback using **sparse sensor configuration**.

<p align="center"><img src="Pics/Project_pipline.png" width="1000" height="600">

It includes five major parts:
- Robot's limb design, manufacturing in mechanical aspect.
- Sensor choice, positioning, assembly, data acquision in mechatronic aspect.
- Data postprocessing in functionality aspect.
- Automatic data collection system in application aspect.
- System integration in robotic application aspect.

## Research agent
We adopt one open-source project ["Poppy Project-Humanoids"](https://www.poppy-project.org/en/) and make contributions to optimize the hardware, software, and web tools.

In this repository, we introduce haptic feedback in robot's limb, which realizes single-contact and **multi-contact** stimulation localization and quantifization functions.

<p align="center"><img src="Pics/Project_object.png" width="800" height="400">

## Robot's limb design, manufacturing in mechanical aspect
### Limb design in [Solidworks](https://www.solidworks.com/de)
- Keep kinematic parameters unchanged
- In Solidworks, [parametrize](http://help.solidworks.com/2017/english/solidworks/cworks/parameters_2.htm) thickness of the flexible sensing shell and the in-middle placed structure support. ![#c5f015](Tutorial follows)

<p align="center"><img src="Pics/Limb_design.png" width="280" height="400" align="center">

### Limb design validation in [ANSYS: Workbench-Static Structure](https://www.ansys.com/products/structures)
- Robustness check, whether the strucutre can support the  whole weight of the robot during its dynamics motion.
- Flexiablity check, how flexible the outer-shell of the limb is agagist stimulation.
- [Material: PA2200](https://www.shapeways.com/rrstatic/material_docs/mds-strongflex.pdf)

<p align="center"><img src="Pics/Limb_design_validation.png" width="280" height="320" align="center">

### 3D printing
- Supplier: [Shapeways](https://www.shapeways.com/)
- Printing material: [White Versatile Plastic](https://www.shapeways.com/materials/versatile-plastic)
  - [Datasheet](https://www.shapeways.com/rrstatic/material_docs/mds-strongflex.pdf)
- Printing approach: [Selective Laser Sintering](https://en.wikipedia.org/wiki/Selective_laser_sintering)

<p align="center"><img src="Pics/Limb_3D_print.png" width="280" height="400" align="center">
  
## Sensor choice, positioning, assembly, data acquision in mechantronic aspect
- Sensor choice: [Strain Gauge:EP-08-250BF-350](http://docs.micro-measurements.com/?id=2573) with high elongation ratio 20% for plastic application.
- Sensor supplier: [Micro Measurement](http://docs.micro-measurements.com)
- Sensor positioning: four methods are compared in [Sun & Martius](https://ieeexplore.ieee.org/abstract/document/8625064)
  - Collection simulation data in [ANSYS](https://www.ansys.com/products/structures):
    - In Workbench: Static Structure -> Engineering data (material elastic properties: Young's modulus, Possio's ratio, Density) 
    - In DesignModeler: Geometry design or import
    - In Mechanics: assign material properties to parts -> mesh properties define -> constrains and force define -> solution define.
    - Automative data collection: iPython in ANSYS Mechanics, Console API (code example for random double nodal forces on surface)
    ``` IPython
    import string, re
    model = ExtAPI.DataModel.Project.Model
    static_structural = model.Analyses[0]
    analysis_settings = static_structural.AnalysisSettings.NumberOfSteps=1 
    path=ExtAPI.DataModel.AnalysisList[0].WorkingDir.Split("\\")
    force_info =  open(string.join(path[:len(path)-2],"\\")+"\\"+"03_TwoForce_Direction.txt")
    a =force_info.read()
    b = a.Split('\n')

    for i in range(3255):
      c = b[i].Split(' ')
      sel = model.AddNamedSelection()
      sel.Name=str(i+1)
      selws = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.MeshNodes)
      selws.Ids =[int(float(c[0]))]
      sel.Location =selws

      force = static_structural.AddNodalForce()
      force.Location = model.NamedSelections.Children[0]
      force.XComponent.Output.DiscreteValues = [Quantity(str(c[1])+" [N]")]
      force.ZComponent.Output.DiscreteValues = [Quantity(str(c[3])+" [N]")]

      sel1 = model.AddNamedSelection()
      sel1.Name=str(i+2)
      selws1 = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.MeshNodes)
      selws1.Ids =[int(float(c[4]))]
      sel1.Location =selws1

      force1 = static_structural.AddNodalForce()
      force1.Location = model.NamedSelections.Children[1]
      force1.XComponent.Output.DiscreteValues = [Quantity(str(c[5])+" [N]")]
      force1.ZComponent.Output.DiscreteValues = [Quantity(str(c[7])+" [N]")]

      solution  = model.Analyses[0].Solution
      total_deformation = solution.AddTotalDeformation()
      static_structural.Solve(True)
      total_deformation.Name = str(i+1)

      total_deformation.ExportToTextFile(True, string.join(path[:len(path)-2],"\\")+"\\"+"two"+str(i+1)+".txt")
      total_deformation.Delete()
            sel.Delete()
      sel1.Delete()	
            force.Delete()
      force1.Delete()
    ```
  - Data-driven methods and Model-based methods are compared. Code and data for those are offered while requested through **huanbo.sunrwth@gmail.com**. Technical explaination is in [Sun & Martius](https://ieeexplore.ieee.org/abstract/document/8625064)
  
  <p align="center"><img src="Pics/Positioning_methods.png" width="800" height="400" align="center">

- Sensor assembly: To assemble strain gauge on curved surface needs a bit of patience and tricks. [M-BOND AE10](https://www.micro-measurements.com/pca/accessories/adhesives) is used as adhesive material to assemble strain gauge on limb internal shell surface. 10 hours are needed for curing, while in-between the strain gauge should be pretightened. Special structure is designed for this purpose (easy in easy out):

 <p align="center"><img src="Pics/Pretighten-Structure.png" width="500" height="400" align="center">




