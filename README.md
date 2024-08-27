# qgis-convertshpgen4-plugin - John Deere Tools

This plugin enables the conversion of spatial data into a format compatible
with the application in new John Deere agricultural machinery monitors.

## Installation

The plugin has not been officially published in the QGIS repository yet, 
so it must be installed manually.

For version [QGIS 3.36 'Maidenheard'](https://qgis.org/en/site/forusers/download.html), it is recommended
to uninstall and reinstall the Fiona library. On Windows open
OSGeo4w Shell with QGIS as Administrator and type:

```bash
pip uninstall fiona
pip install fiona
```

In Linux or macOS, you can install the library using the Terminal, 
as QGIS uses the standard Python installation. 

To add the plugin to QGIS 3.3 manually, you can follow these steps:

1. Download plugin repository from GitHub: https://github.com/sirpetr/qgis-convertshpgen4-plugin.git
2. Open QGIS 3.3x and go to ```Plugins``` -> ```Manage and Install Plugins``` -> ```Install from ZIP```
and browse select ZIP file and ```Install Plugin```
3. In ```Manage and Install Plugins``` -> ```Installed``` make sure if is checked John Deere Tools

A new icon for John Deere Tools will be added to the Vector menu and the QGIS main panel.

![](data/vector.png)

## Tool Convert Shapefiles to Gen4 and example requirements

This tool allows the conversion of shapefiles (polygons or polylines) into a Gen4 folder
containing XML and JSON files for import into John Deere navigation monitors or 
the Operation Center.

Polygons must correspond to the boundaries of field and their attributes must correspond to the columns with the names 
of clients, farms, fields and crops. Guidance polylines must be located within the boundaries of the given fields.

The column with guidance line type categories must contain numbers corresponding to the following: 
**<span style="color:red">1 for AB Line</span>**, **<span style="color:red">2 for 
AB Curve</span>**, and **<span style="color:red">3 for Adaptive Curve</span>**. If the number in the column is less 
than 1 or greater than 3, it will automatically be set to 3, which corresponds to Adaptive Curve. Additionally, 
if this parameter is not filled in, it defaults to Adaptive Curve.

![](data/input_line_type.png)

To run the practical example, you need an input shapefile layer with polygons and an optional shapefile layer with 
polylines. 
Both layers should be in the WGS 1984 coordinate system (EPSG:4326). 
The next parameters are from attribute table from layer or custom text:

![](data/input.png)

- **Column with <span style="color:red">client</span> names or custom name** - required parameter.
- **Column with <span style="color:blue">farm</span> names or custom name** - required parameter.
- **Column with <span style="color:yellow">field</span> names** - required parameter from attribute 
table in polygon layer. In the column, rows have to be unique values.
- **Column with <span style="color:green">crop</span> names** - not required parameter
- **Column with guidance line type categories** - not required parameter from 
attribute. If it will be empty then Adaptive Curve is default.
- **Column with guidance line names** - not required parameter from attribute table in line layer. In the column, 
rows have to be unique values. 

![](data/output.png)

![](data/output_2.png)

The resulting Gen4 folder will be compressed into a ZIP archive ready 
for import into the Operations Center or for further distribution.
Important piece of information is that it is possible to import the same 
farms, fields, and boundaries under different clients into the Operations Center.

![](data/output_jd1.png)

![](data/output_jd2.png)

## License

This QGIS plugin is under the GNU General Public License (GPL). This open-source
license allows users to freely use, modify, and distribute the software, provided that any derivative works also 
remain under the GPL. 

## Contact 
Petr Sirucek: [petr.siruck@gmail.com]()