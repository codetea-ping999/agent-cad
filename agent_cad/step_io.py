"""STEP AP242 export with product structure, names, colours, material and
geometric validation properties (volume / area / centroid), plus reader helpers.

Validation properties let a receiving CAD (e.g. CATIA) confirm on import that
the geometry it rebuilt matches what was sent.
"""
import re
from dataclasses import dataclass
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString, TCollection_HAsciiString, TCollection_AsciiString
from OCP.XCAFDoc import (XCAFDoc, XCAFDoc_ShapeTool, XCAFDoc_ColorTool, XCAFDoc_DocumentTool, XCAFDoc_Volume, XCAFDoc_Area,
                         XCAFDoc_Centroid, XCAFDoc_ColorType, XCAFDoc_Material, XCAFDoc_MaterialTool)
from OCP.TDataStd import TDataStd_TreeNode
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
DEFAULT_ASSEMBLY = "ASSEMBLY"   # name given to a multi-part export when none is passed


@dataclass
class PartSpec:
    name: str
    shape: Shape
    color: tuple = (0.6, 0.64, 0.7)       # RGB 0..1
    material: str = ""
    density_g_cm3: float = 0.0
    pmi: list | None = None                # drawing dimensions to write as semantic PMI (agent_cad.pmi)
    pmi_skipped: list | None = None        # set by write_step: those not linked to faces


def definitions(parts):
    """Group parts into shared definitions: the same B-rep with identical metadata
    (e.g. repeated fasteners) is one product with several instances, wherever each
    instance is placed (the location is not part of the definition). Returns
    [(PartSpec, [indices])] in first-occurrence order."""
    out = []
    for i, ps in enumerate(parts):
        for d, idx in out:
            a, b = ps.shape.wrapped, d.shape.wrapped        # same B-rep and orientation, any location
            if a.IsPartner(b) and a.Orientation() == b.Orientation() and (ps.name, tuple(ps.color), ps.material, ps.density_g_cm3) == \
                    (d.name, tuple(d.color), d.material, d.density_g_cm3):
                idx.append(i); break
        else:
            out.append((ps, [i]))
    return out


def mass_props(shape, eps=1e-9):
    """Volume, area, centroid with adaptive integration. The default (fixed
    Gauss order) integration scatters by ~1e-5 on B-spline faces, which would
    mask real exchange errors."""
    v, a = GProp_GProps(), GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, v, Eps=eps)   # the Eps overload = adaptive Gauss integration
    BRepGProp.SurfaceProperties_s(shape, a, Eps=eps)
    return v.Mass(), a.Mass(), v.CentreOfMass()


def _new_doc():
    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("XmlXCAF"))
    app.InitDocument(doc)
    return doc


def _flatten(shp):
    """A compound made only of (nested, located) compounds and solids -> its solids,
    each with its placement baked into the geometry. A nested location the writer
    cannot express as a rigid placement (a mirror) otherwise turns the part into an
    unnamed sub-product and its material is lost. Anything else is left untouched,
    so the exchange check still sees stray faces or edges."""
    from OCP.TopoDS import TopoDS_Iterator, TopoDS_Compound
    from OCP.TopAbs import TopAbs_COMPOUND, TopAbs_SOLID
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    if shp.ShapeType() != TopAbs_COMPOUND:
        return shp
    solids, nested = [], False

    def walk(s, loc):
        nonlocal nested
        it = TopoDS_Iterator(s, False, False)
        while it.More():
            c = it.Value()
            cl = loc.Multiplied(c.Location())
            if c.ShapeType() == TopAbs_COMPOUND:
                nested = True
                if not walk(c, cl):
                    return False
            elif c.ShapeType() == TopAbs_SOLID:
                solids.append((c, cl))
            else:
                return False
            it.Next()
        return True
    if not walk(shp, TopLoc_Location()) or not nested or not solids:
        return shp
    baked = [c.Located(TopLoc_Location()) if l.IsIdentity() else
             BRepBuilderAPI_Transform(c.Located(TopLoc_Location()), l.Transformation(), True).Shape()
             for c, l in solids]
    if len(baked) == 1:
        return baked[0]
    out = TopoDS_Compound(); bb = BRep_Builder(); bb.MakeCompound(out)
    for c in baked:
        bb.Add(out, c)
    return out


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
        TDataStd_Name.Set_s(asm, TCollection_ExtendedString(assembly_name or DEFAULT_ASSEMBLY))
    labels, comp = [], [None] * len(parts)
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy, BRepBuilderAPI_Transform
    for defn, idx in definitions(parts):
        # the definition is the unplaced B-rep; each instance's placement goes on its component
        loc = defn.shape.wrapped.Location()
        shp = _flatten(defn.shape.wrapped.Located(TopLoc_Location()))
        if asm is None and not loc.IsIdentity():   # bare part: no component to carry the placement
            shp = BRepBuilderAPI_Transform(shp, loc.Transformation(), True).Shape()
        if any(shp.IsPartner(b) for _, b in labels):
            # same B-rep, different metadata: AddShape would reuse (and overwrite) the label
            shp = BRepBuilderAPI_Copy(shp).Shape()
        ps = defn
        lab = st.AddShape(shp, False)
        labels.append((lab, shp))
        for i in idx:
            comp[i] = (lab, parts[i].shape.wrapped.Location())
        TDataStd_Name.Set_s(lab, TCollection_ExtendedString(ps.name))
        r, g, b = ps.color
        ct.SetColor(lab, Quantity_Color(r, g, b, Quantity_TOC_RGB), XCAFDoc_ColorType.XCAFDoc_ColorSurf)
        vol, area, cog = mass_props(shp)              # in the definition's own frame
        XCAFDoc_Volume.Set_s(lab, vol)
        XCAFDoc_Area.Set_s(lab, area)
        XCAFDoc_Centroid.Set_s(lab, cog)
        if ps.material:
            h = lambda s: TCollection_HAsciiString(s)
            m = mt.AddMaterial(h(ps.material), h(ps.material), ps.density_g_cm3,  # written as g/cm3
                               h("density"), h("POSITIVE_RATIO_MEASURE"))
            mt.SetMaterial(lab, m)
        if ps.pmi:
            from .pmi import attach
            ps.pmi_skipped = attach(doc, lab, shp, ps.pmi)
    if asm is not None:
        for lab, loc in comp:                 # one component per part, in the caller's order
            st.AddComponent(asm, lab, loc)
    st.UpdateAssemblies()
    w = STEPCAFControl_Writer()  # constructor initialises the STEP statics; set ours afterwards
    Interface_Static.SetIVal_s("write.step.schema", AP242)
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    w.SetNameMode(True); w.SetColorMode(True); w.SetPropsMode(True); w.SetMaterialMode(True)
    w.SetDimTolMode(any(ps.pmi for ps in parts))
    if not w.Transfer(doc, STEPControl_AsIs):
        raise IOError(f"STEP transfer failed (nothing written): {path}")
    from OCP.APIHeaderSection import APIHeaderSection_MakeHeader
    mh = APIHeaderSection_MakeHeader(w.ChangeWriter().Model())
    mh.SetName(TCollection_HAsciiString(re.split(r"[\\/]", str(path))[-1]))  # file name only, never the directory
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
    material: str | None = None
    density_g_cm3: float | None = None
    n_roots: int = 1  # top-level products in the file (a valid export has exactly one)
    path: tuple = ()  # assembly names from the root down to this part's parent


def read_step(path):
    """Read a STEP file with XCAF: returns (assembly_name|None, [ReadPart]).
    Lengths are converted to mm whatever unit the file declares."""
    doc = _new_doc()
    XCAFDoc_DocumentTool.SetLengthUnit_s(doc, 0.001)  # convert whatever the file uses to mm
    r = STEPCAFControl_Reader()
    r.SetNameMode(True); r.SetColorMode(True); r.SetPropsMode(True); r.SetMatMode(True)
    if r.ReadFile(str(path)) != IFSelect_RetDone:
        raise IOError(f"STEP read failed: {path}")
    if not r.Transfer(doc):
        raise IOError(f"STEP transfer failed (model not fully imported): {path}")
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

    def occurrence_color(shp):
        c = Quantity_Color()
        for t in (XCAFDoc_ColorType.XCAFDoc_ColorSurf, XCAFDoc_ColorType.XCAFDoc_ColorGen):
            if ct.GetInstanceColor(shp, t, c):
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

    def material(lab):
        # IsAttribute gates FindAttribute (see above); a material with zero density is still a material
        if not lab.IsAttribute(XCAFDoc.MaterialRefGUID_s()):
            return None, None
        tn = TDataStd_TreeNode(); lab.FindAttribute(XCAFDoc.MaterialRefGUID_s(), tn)
        if not tn.HasFather() or not tn.Father().Label().IsAttribute(XCAFDoc_Material.GetID_s()):
            return None, None
        m = XCAFDoc_Material(); tn.Father().Label().FindAttribute(XCAFDoc_Material.GetID_s(), m)
        return m.GetName().ToCString(), m.GetDensity()  # density as written (g/cm3)

    free = TDF_LabelSequence(); st.GetFreeShapes(free)
    asm_name, parts = None, []

    def walk(assembly, placed, path):
        """Leaf parts of an assembly, through any depth of subassemblies, each placed
        by the product of the component locations above it."""
        comps = TDF_LabelSequence(); st.GetComponents_s(assembly, comps, False)
        for j in range(1, comps.Length() + 1):
            comp = comps.Value(j)
            ref = TDF_Label(); st.GetReferredShape_s(comp, ref)
            loc = placed.Multiplied(XCAFDoc_ShapeTool.GetLocation_s(comp))
            if st.IsAssembly_s(ref):
                walk(ref, loc, path + (name(ref),))
                continue
            base = st.GetShape_s(ref)
            shp = base.Located(loc.Multiplied(base.Location()))
            val = validation(ref) or validation(comp)
            if "centroid" in val:               # stored in the definition's frame: place it like the instance
                c = gp_Pnt(*val["centroid"]).Transformed(loc.Transformation())
                val["centroid"] = (c.X(), c.Y(), c.Z())
            # an occurrence may carry its own colour, overriding its definition's
            parts.append(ReadPart(name(ref), Shape.cast(shp), occurrence_color(shp) or color(comp) or color(ref), val, *material(ref),
                                  path=path))

    for i in range(1, free.Length() + 1):
        top = free.Value(i)
        if st.IsAssembly_s(top):
            asm_name = name(top)
            walk(top, TopLoc_Location(), (asm_name,))
        else:
            parts.append(ReadPart(name(top), Shape.cast(st.GetShape_s(top)), color(top), validation(top), *material(top)))
    for rp in parts:
        rp.n_roots = free.Length()
    return asm_name, parts
