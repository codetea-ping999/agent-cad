"""STEP AP242 export with product structure, names, colours, material and
geometric validation properties (volume / area / centroid), plus reader helpers.

Validation properties let a receiving CAD (e.g. CATIA) confirm on import that
the geometry it rebuilt matches what was sent.
"""
from dataclasses import dataclass
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString, TCollection_HAsciiString, TCollection_AsciiString
from OCP.XCAFDoc import XCAFDoc_ColorTool, XCAFDoc_DocumentTool, XCAFDoc_Volume, XCAFDoc_Area, XCAFDoc_Centroid, XCAFDoc_ColorType
from OCP.XCAFApp import XCAFApp_Application
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label
from OCP.collections import Sequence_TDF_Label as TDF_LabelSequence
from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCP.STEPCAFControl import STEPCAFControl_Writer, STEPCAFControl_Reader
from OCP.STEPControl import STEPControl_AsIs
from OCP.Interface import Interface_Static
from OCP.IFSelect import IFSelect_RetDone
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.TopLoc import TopLoc_Location
from OCP.gp import gp_Pnt
from build123d import Shape

AP242 = 5  # write.step.schema value for AP242DIS


@dataclass
class PartSpec:
    name: str
    shape: Shape
    color: tuple = (0.6, 0.64, 0.7)       # RGB 0..1
    material: str = ""
    density_g_cm3: float = 0.0


def mass_props(shape, eps=1e-9):
    """Volume, area, centroid with adaptive integration. The default (fixed
    Gauss order) integration scatters by ~1e-5 on B-spline faces, which would
    mask real exchange errors."""
    v, a = GProp_GProps(), GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, v, eps)
    BRepGProp.SurfaceProperties_s(shape, a, eps)
    return v.Mass(), a.Mass(), v.CentreOfMass()


def _new_doc():
    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("XmlXCAF"))
    app.InitDocument(doc)
    return doc


def write_step(parts, path, assembly_name=None, author="agent-cad", organization="agent-cad"):
    """Write one or more parts as AP242. Several parts -> an assembly product
    with one named sub-product per part. Units: millimetre."""
    doc = _new_doc()
    main = doc.Main()
    st = XCAFDoc_DocumentTool.ShapeTool_s(main)
    ct = XCAFDoc_DocumentTool.ColorTool_s(main)
    mt = XCAFDoc_DocumentTool.MaterialTool_s(main)
    XCAFDoc_DocumentTool.SetLengthUnit_s(doc, 0.001)  # mm (value is in metres)
    asm = None
    if len(parts) > 1 or assembly_name:
        asm = st.NewShape()
        TDataStd_Name.Set_s(asm, TCollection_ExtendedString(assembly_name or "ASSEMBLY"))
    for ps in parts:
        lab = st.AddShape(ps.shape.wrapped, False)
        TDataStd_Name.Set_s(lab, TCollection_ExtendedString(ps.name))
        r, g, b = ps.color
        ct.SetColor(lab, Quantity_Color(r, g, b, Quantity_TOC_RGB), XCAFDoc_ColorType.XCAFDoc_ColorSurf)
        vol, area, cog = mass_props(ps.shape.wrapped)
        XCAFDoc_Volume.Set_s(lab, vol)
        XCAFDoc_Area.Set_s(lab, area)
        XCAFDoc_Centroid.Set_s(lab, cog)
        if ps.material:
            h = lambda s: TCollection_HAsciiString(s)
            m = mt.AddMaterial(h(ps.material), h(ps.material), ps.density_g_cm3,  # written as g/cm3
                               h("density"), h("POSITIVE_RATIO_MEASURE"))
            mt.SetMaterial(lab, m)
        if asm is not None:
            st.AddComponent(asm, lab, TopLoc_Location())
    st.UpdateAssemblies()
    w = STEPCAFControl_Writer()  # constructor initialises the STEP statics; set ours afterwards
    Interface_Static.SetIVal_s("write.step.schema", AP242)
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    w.SetNameMode(True); w.SetColorMode(True); w.SetPropsMode(True); w.SetMaterialMode(True)
    w.Transfer(doc, STEPControl_AsIs)
    from OCP.APIHeaderSection import APIHeaderSection_MakeHeader
    mh = APIHeaderSection_MakeHeader(w.ChangeWriter().Model())
    mh.SetName(TCollection_HAsciiString(str(path).split("/")[-1]))
    mh.SetAuthorValue(1, TCollection_HAsciiString(author))
    mh.SetOrganizationValue(1, TCollection_HAsciiString(organization))
    mh.SetOriginatingSystem(TCollection_HAsciiString("agent-cad / OpenCascade"))
    if w.Write(str(path)) != IFSelect_RetDone:
        raise IOError(f"STEP write failed: {path}")
    return path


@dataclass
class ReadPart:
    name: str
    shape: Shape
    color: tuple | None
    validation: dict  # props stored in the file (may be empty)


def read_step(path):
    """Read a STEP file with XCAF: returns (assembly_name|None, [ReadPart], length_unit_mm)."""
    doc = _new_doc()
    XCAFDoc_DocumentTool.SetLengthUnit_s(doc, 0.001)  # convert whatever the file uses to mm
    r = STEPCAFControl_Reader()
    r.SetNameMode(True); r.SetColorMode(True); r.SetPropsMode(True)
    if r.ReadFile(str(path)) != IFSelect_RetDone:
        raise IOError(f"STEP read failed: {path}")
    r.Transfer(doc)
    main = doc.Main()
    st = XCAFDoc_DocumentTool.ShapeTool_s(main)
    ct = XCAFDoc_DocumentTool.ColorTool_s(main)

    def name(lab):
        n = TDataStd_Name()
        if lab.FindAttribute(TDataStd_Name.GetID_s(), n):
            return n.Get().ToExtString()
        return ""

    def color(lab):
        c = Quantity_Color()
        for t in (XCAFDoc_ColorType.XCAFDoc_ColorSurf, XCAFDoc_ColorType.XCAFDoc_ColorGen):
            if XCAFDoc_ColorTool.GetColor_s(lab, t, c):
                return (round(c.Red(), 3), round(c.Green(), 3), round(c.Blue(), 3))
        return None

    def validation(lab):
        # Get_s gates the lookup: FindAttribute on a label without the
        # attribute crashes in OCP.
        out = {}
        for key, cls in (("volume", XCAFDoc_Volume), ("area", XCAFDoc_Area)):
            if cls.Get_s(lab, 0.0):
                a = cls(); lab.FindAttribute(cls.GetID_s(), a); out[key] = a.Get()
        p = gp_Pnt()
        if XCAFDoc_Centroid.Get_s(lab, p):
            out["centroid"] = (p.X(), p.Y(), p.Z())
        return out

    free = TDF_LabelSequence(); st.GetFreeShapes(free)
    asm_name, parts = None, []
    for i in range(1, free.Length() + 1):
        top = free.Value(i)
        if st.IsAssembly_s(top):
            asm_name = name(top)
            comps = TDF_LabelSequence(); st.GetComponents_s(top, comps, False)
            for j in range(1, comps.Length() + 1):
                ref = TDF_Label(); st.GetReferredShape_s(comps.Value(j), ref)
                shp = st.GetShape_s(comps.Value(j))
                parts.append(ReadPart(name(ref), Shape.cast(shp), color(ref), validation(ref) or validation(comps.Value(j))))
        else:
            parts.append(ReadPart(name(top), Shape.cast(st.GetShape_s(top)), color(top), validation(top)))
    return asm_name, parts
