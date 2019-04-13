# Collection of simulation data in a automatic manner with [ANSYS](https://www.ansys.com/products/structures):
- In Workbench: Static Structure -> Engineering data (material elastic properties: Young's modulus, Possio's ratio, Density)

<p align="center"><img src="Pics/Data_Collection_SS.png" width="800" height="400">
<p align="center"><img src="Pics/Data_Collection_ED.png" width="800" height="400">

- In Geometry: (SpaceClaim) Geometry design or import
<p align="center"><img src="Pics/Data_Collection_GM.png" width="800" height="400">
  
- In Mechanics: assign material properties to parts -> mesh properties define -> constrains and force define -> solution define.
<p align="center"><img src="Pics/Data_Collection_M.png" width="800" height="400">
  
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
