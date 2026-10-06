# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import re
import warnings
import textwrap
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Rectangle, Polygon, ConnectionPatch
from matplotlib.ticker import FuncFormatter, MultipleLocator
from shapely.geometry import mapping, box as sbox
import streamlit as st

warnings.filterwarnings("ignore")

try:
    import contextily as cx
except ImportError:
    cx = None

# ==============================================================================
# STREAMLIT UI CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="Study Area Map Generator", layout="wide")
st.title("Geospatial Study Area Map Generator")
st.markdown("Generate publication-ready 3-panel study area maps using Google Earth Engine and OSM.")

st.sidebar.header("1. Country & Admin Data")
COUNTRY = st.sidebar.text_input("Country", "Bangladesh")
DISTRICT_NAME = st.sidebar.text_input("District Name", "Rajshahi")
AOI_NAME = st.sidebar.text_input("Study Area (Upazila)", "Charghat")

st.sidebar.header("2. Map Content")
PANEL_C_LAYER = st.sidebar.selectbox("Panel (c) Layer", ["DEM", "OSM", "NONE"])
BASEMAP_A = st.sidebar.selectbox("Panel (a) Basemap", ["topo", "imagery", "street", "light"], index=0)
BASEMAP_B = st.sidebar.selectbox("Panel (b) Basemap", ["topo", "imagery", "street", "light"], index=0)
BASEMAP_C = st.sidebar.selectbox("Panel (c) Basemap", ["topo", "imagery", "street", "light"], index=0)

st.sidebar.header("3. Advanced Settings")
DEM_SCALE_M = st.sidebar.number_input("DEM Scale (m)", value=30, step=10)
SHOW_BASEMAP_A = st.sidebar.checkbox("Show Basemap in Panel (a)", value=True)
SHOW_BASEMAP_B = st.sidebar.checkbox("Show Basemap in Panel (b)", value=True)
SHOW_BASEMAP_C = st.sidebar.checkbox("Show Basemap in Panel (c)", value=True)

# Fixed variables based on your script
DISTRICTS_SOURCE = "FAO/GAUL/2015/level2"
COUNTRY_SOURCE   = "FAO/GAUL/2015/level0"
UPAZILA_SOURCE   = "projects/geo-gee-imtiaj/assets/BGD_adm3_Upazilla"
F_COUNTRY, F_DIVISION, F_DISTRICT = "ADM0_NAME", "ADM1_NAME", "ADM2_NAME"
AOI_MODE = "NAME"
AOI_SOURCE = "projects/geo-gee-imtiaj/assets/BGD_adm3_Upazilla"
AOI_NAME_FIELD = "ADM3_NAME"
DISTRICTS = [DISTRICT_NAME]
AOI_DISPLAY_NAME = f"{AOI_NAME} Upazila"
AOI_SUFFIX = None
PANEL_B_EXTENT = "DISTRICT"
TITLE_A_OVERRIDE = TITLE_B_OVERRIDE = TITLE_C_OVERRIDE = CAPTION_OVERRIDE = None

SHOW_CONTEXT_A = True
NEIGHBOUR_SOURCE = "FAO/GAUL/2015/level0"
CONTEXT_PAD_DEG = 7
SHOW_CONTEXT_LABELS = True
SEA_LABEL = "Bay of Bengal"
BASEMAP_A_ALPHA = BASEMAP_B_ALPHA = 0.8
DISTRICT_FILL_ALPHA = 0.55
BASEMAP_ALPHA = 0.7
BASEMAP_ZOOM = None
DEM_CMAP, DEM_ALPHA = "viridis", 0.95

OSM_LAYERS = dict(water=True, rivers=True, buildings=True, roads=True, rail=True)
OSM_RIVER_TYPES = ["river", "canal"]
OSM_ROAD_EXCLUDE = ["footway", "path", "steps", "pedestrian", "cycleway", "bridleway", "corridor", "proposed", "construction", "track"]
OSM_MAJOR_ROADS  = ["motorway", "trunk", "primary", "secondary", "motorway_link", "trunk_link", "primary_link", "secondary_link"]
OSM_USE_CACHE = True
OSM_STYLE = dict(
    water     = dict(color="#A9D3F5", label="Water body"),
    rivers    = dict(color="#3C8DD8", lw=1.0, label="River / canal"),
    buildings = dict(color="#8A8A8A", label="Building"),
    roads     = dict(color="#4D4D4D", lw_major=1.4, lw_minor=0.5, label="Road"),
    rail      = dict(color="black", lw=1.8, label="Railway"),
)
LEGEND_C_LOC = "lower left"
LEGEND_ROW_PAD = 0.03

COL_COUNTRY_FILL, COL_COUNTRY_LINE, COL_COUNTRY_EDGE = "white", "#BDBDBD", "#555555"
COL_DISTRICT_FILL, COL_DISTRICT_EDGE, COL_FOCAL_EDGE = "#FDE8C0", "#9A9A9A", "#555555"
COL_UPAZILA_LINE, COL_AOI, COL_LOCATOR, COL_FRAME = "#B0A58F", "#E41A1C", "#E41A1C", "black"
COL_SEA, SEA_ALPHA = "#8EC3E8", 0.35
COL_NEIGH_FILL, NEIGH_ALPHA, COL_NEIGH_EDGE = "#CFCFCF", 0.55, "#9A9A9A"
COL_CTX_LABEL, COL_SEA_LABEL = "#6B6B6B", "#2F6EA3"
AOI_LW_B, AOI_LW_C = 1.6, 2.0

FIG_SIZE = (11.5, 8.2)
FIG_LEFT, FIG_RIGHT, FIG_BOTTOM, FIG_TOP = 0.02, 0.98, 0.115, 0.985
MID_X, MID_Y, GAP_X, GAP_Y = 0.40, 0.55, 0.008, 0.012
INNER_A, INNER_B, INNER_C = [0.15, 0.115, 0.80, 0.85], [0.15, 0.115, 0.80, 0.85], [0.105, 0.155, 0.865, 0.815]
PAD_A, PAD_B, PAD_C = (0.40, 0.40, 0.38, 0.16), (0.12, 0.12, 0.12, 0.20), (0.10, 0.10, 0.10, 0.16)

FONT_FAMILY = "DejaVu Serif"
FONT_TITLE, FONT_MAIN, FONT_TICK, FONT_LEGEND = 12, 9, 7, 8
LW_OUTER, LW_MAP = 1.2, 1.0

FMT_A, FMT_B, FMT_C = "deg", "dm", "dm"
NBINS_A, NBINS_B, NBINS_C = (4, 5), (3, 4), (4, 5)
HEMI = dict(a=False, b=False, c=True)

SHOW_PANEL_BOXES = False
SHOW_LOCATOR = True
SHOW_NORTH_ARROW = dict(a=False, b=False, c=True)
SHOW_SCALEBAR = dict(a=False, b=False, c=True)
SHOW_LEGEND = dict(a=False, b=False, c=True)
SHOW_CAPTION = True
SCALEBAR_C_POS, COLORBAR_C_POS = (0.64, -0.125), [0.06, -0.125, 0.34, 0.016]

OUTPUT_DIR = "Final_Revised_Figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# CACHED GEE INITIALIZATION & DATA HELPERS
# ==============================================================================
@st.cache_resource
def init_ee():
    import ee
    import geemap
    import google.oauth2.credentials
    
    try:
        # 1. Attempt to load from Streamlit Cloud Secrets (for deployment)
        refresh_token = st.secrets["gee"]["refresh_token"]
        project = st.secrets["gee"]["project"]
        
        # Earth Engine's default OAuth client credentials
        client_id = "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com"
        client_secret = "d-qW80SyliBiMeTlW9WIN25K"
        
        creds = google.oauth2.credentials.Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret,
            scopes=["https://www.googleapis.com/auth/earthengine", "https://www.googleapis.com/auth/cloud-platform"]
        )
        ee.Initialize(credentials=creds, project=project)
        
    except Exception as e:
        # 2. Fallback for your local machine
        try:
            ee.Initialize(project='geo-gee-imtiaj')
        except Exception:
            ee.Authenticate()
            ee.Initialize(project='geo-gee-imtiaj')
            
    return ee, geemap

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if st.sidebar.button("Generate Map"):
    with st.spinner("Initializing GEE and fetching data..."):
        ee, geemap = init_ee()
        
        NAME_FIELDS = ["NAME_3", "ADM3_NAME", "NAME_2", "ADM2_NAME", "NAME", "name", "Name", "UPAZILA", "Upazila", "upazila", "NAME_EN", "ADM_NAME"]

        def read_vector(src, label, ee_filter=None, bbox=None):
            if os.path.exists(str(src)):
                g = gpd.read_file(src)
            else:
                fc = ee.FeatureCollection(src)
                if ee_filter is not None: fc = ee_filter(fc)
                if bbox is not None: fc = fc.filterBounds(ee.Geometry.Rectangle([float(v) for v in bbox]))
                g = geemap.ee_to_gdf(fc)
            return g.set_crs(4326) if g.crs is None else g.to_crs(4326)

        def read_neighbours(src, bbox):
            if os.path.exists(str(src)):
                g = gpd.read_file(src)
                g = g.set_crs(4326) if g.crs is None else g.to_crs(4326)
                return gpd.clip(g, sbox(*bbox))
            rect = ee.Geometry.Rectangle([float(v) for v in bbox])
            fc = ee.FeatureCollection(src).filterBounds(rect)
            fc = fc.map(lambda f: ee.Feature(f.geometry().intersection(rect, 1000).simplify(1000)).copyProperties(f))
            g = geemap.ee_to_gdf(fc)
            return g.set_crs(4326) if g.crs is None else g.to_crs(4326)

        def union(g): return g.geometry.union_all() if hasattr(g.geometry, "union_all") else g.geometry.unary_union
        def join_names(names): names = [str(n) for n in names]; return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " & " + names[-1]
        def find_field(gdf, preferred=None):
            if preferred and preferred in gdf.columns: return preferred
            for c in NAME_FIELDS:
                if c in gdf.columns: return c
            return None
        def select_by_names(gdf, field, names, what):
            names = [names] if isinstance(names, str) else list(names)
            key = gdf[field].astype(str).str.strip().str.lower()
            want = [n.strip().lower() for n in names]
            return gdf[key.isin(want)]
        def only_country(g):
            if F_COUNTRY in g.columns: g = g[g[F_COUNTRY].astype(str).str.lower() == COUNTRY.lower()]
            return g

        _cf = (lambda fc: fc.filter(ee.Filter.eq(F_COUNTRY, COUNTRY)))
        districts_all = only_country(read_vector(DISTRICTS_SOURCE, "Districts", _cf))
        country_gdf = only_country(read_vector(COUNTRY_SOURCE, "Country", _cf)) if COUNTRY_SOURCE else districts_all.dissolve()
        country_union = union(country_gdf)

        neigh_gdf = None
        if SHOW_CONTEXT_A:
            try:
                cb = country_gdf.total_bounds
                neigh_gdf = read_neighbours(NEIGHBOUR_SOURCE, (cb[0] - CONTEXT_PAD_DEG, cb[1] - CONTEXT_PAD_DEG, cb[2] + CONTEXT_PAD_DEG, cb[3] + CONTEXT_PAD_DEG))
                if F_COUNTRY in neigh_gdf.columns: neigh_gdf = neigh_gdf[neigh_gdf[F_COUNTRY].astype(str).str.lower() != COUNTRY.lower()]
            except: pass

        focal = select_by_names(districts_all, F_DISTRICT, DISTRICTS, "District") if AOI_MODE == "DISTRICTS" else None
        
        if AOI_MODE in ("TABLE_ID", "NAME"):
            raw = read_vector(AOI_SOURCE, "AOI table")
            fld = find_field(raw, AOI_NAME_FIELD)
            raw = select_by_names(raw, fld, AOI_NAME, "AOI")
            aoi_name = join_names([AOI_NAME] if isinstance(AOI_NAME, str) else AOI_NAME)
            aoi_union = union(raw)
            auto_suffix = ""

        aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_union], crs=4326)
        aoi_label = AOI_DISPLAY_NAME or f"{aoi_name} {auto_suffix}".strip()

        if focal is None:
            focal = select_by_names(districts_all, F_DISTRICT, DISTRICT_NAME, "District")
        
        context_gdf = focal
        fn = list(focal[F_DISTRICT])
        b_name = f"{join_names(fn)} District" + ("s" if len(fn) > 1 else "")
        context_union = union(context_gdf)
        b_bounds = context_union.bounds

        upazilas_gdf = None
        if UPAZILA_SOURCE:
            try:
                up = read_vector(UPAZILA_SOURCE, "Upazilas", bbox=b_bounds)
                upazilas_gdf = up[up.representative_point().within(context_union)]
            except: pass

        TITLE_A, TITLE_B, TITLE_C = COUNTRY, b_name, aoi_label
        LEGEND_C = f"{aoi_label} Boundary"
        CAPTION = f"Figure 1. Location of (a) {b_name} in {COUNTRY}; (b) {aoi_label} in {b_name}; and (c) base map of {aoi_label}."
        
        dem_arr = vmin = vmax = None
        if PANEL_C_LAYER == "DEM":
            with st.spinner("Auto-scaling and fetching DEM payload..."):
                geom = ee.Geometry(mapping(aoi_union))
                dem = ee.Image("USGS/SRTMGL1_003").rename("elevation")
                mask_band = ee.Image(1).clip(geom).unmask(0).rename("ROI_MASK")
                comb = dem.addBands(mask_band).unmask(-9999)
                current_scale = DEM_SCALE_M
                while True:
                    try:
                        arr = geemap.ee_to_numpy(comb, region=geom, scale=current_scale).astype(float)
                        break
                    except:
                        current_scale += 30
                dem_arr = np.where(arr[:, :, 1] == 1, arr[:, :, 0], np.nan)
                dem_arr[dem_arr < -9000] = np.nan
                v = dem_arr[~np.isnan(dem_arr)]
                vmin, vmax = np.percentile(v, 2), np.percentile(v, 98)

        osm_data = {}
        if PANEL_C_LAYER == "OSM":
            with st.spinner("Fetching OSM features..."):
                import osmnx as ox
                ox.settings.use_cache = True
                LINES, POLYS = ["LineString", "MultiLineString"], ["Polygon", "MultiPolygon"]
                
                def load_osm_layer(name, tags, kinds, post=None):
                    try: g = ox.features_from_polygon(aoi_union, tags).reset_index(drop=True)
                    except: return None
                    g = g[g.geom_type.isin(kinds)]
                    keep = [c for c in ("highway", "waterway") if c in g.columns]
                    g = g[keep + ["geometry"]].copy()
                    for c in keep: g[c] = g[c].astype(str)
                    if post is not None: g = post(g)
                    if kinds is POLYS: g["geometry"] = g.geometry.buffer(0)
                    g = gpd.clip(g, aoi_gdf)
                    return g[~g.geometry.is_empty & g.geometry.notna()]

                if OSM_LAYERS.get("water"): osm_data["water"] = load_osm_layer("water", {"natural": "water", "waterway": "riverbank"}, POLYS)
                if OSM_LAYERS.get("rivers"): osm_data["rivers"] = load_osm_layer("rivers", {"waterway": OSM_RIVER_TYPES}, LINES)
                if OSM_LAYERS.get("buildings"): osm_data["buildings"] = load_osm_layer("buildings", {"building": True}, POLYS)
                if OSM_LAYERS.get("roads"): osm_data["roads"] = load_osm_layer("roads", {"highway": True}, LINES, post=lambda g: g[~g["highway"].isin(OSM_ROAD_EXCLUDE)])
                if OSM_LAYERS.get("rail"): osm_data["rail"] = load_osm_layer("rail", {"railway": "rail"}, LINES)

        # ---------------- CARTOGRAPHY ----------------
        with st.spinner("Rendering Cartography..."):
            plt.rcParams.update({"font.family": FONT_FAMILY})
            fig = plt.figure(figsize=FIG_SIZE, facecolor="white")

            def sub_rect(box, m): return [box[0] + m[0] * box[2], box[1] + m[1] * box[3], m[2] * box[2], m[3] * box[3]]
            def rect(x0, y0, x1, y1): return [x0, y0, x1 - x0, y1 - y0]
            
            def make_map(fig, box, margin, bounds, title, pad, fmt, nbins, key):
                ax = fig.add_axes(sub_rect(box, margin))
                x0, y0, x1, y1 = bounds
                w, h = x1 - x0, y1 - y0
                ax.set_xlim(x0 - w * pad[0], x1 + w * pad[1])
                ax.set_ylim(y0 - h * pad[2], y1 + h * pad[3])
                aspect = 1 / np.cos(np.radians((y0 + y1) / 2))
                ax.set_aspect(aspect, adjustable="datalim")
                ax.apply_aspect()
                ax.tick_params(axis="both", direction="out", length=3, width=0.7, pad=2, labelsize=FONT_TICK, top=False, right=False)
                ax.tick_params(axis="y", labelrotation=90)
                for sp in ax.spines.values(): sp.set_linewidth(LW_MAP); sp.set_edgecolor(COL_FRAME)
                bb = dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5)
                ax.text(0.03, 0.975, f"({key})", transform=ax.transAxes, ha="left", va="top", fontsize=FONT_TITLE, fontweight="bold", bbox=bb, zorder=100)
                ax.text(0.55, 0.975, textwrap.fill(title, 26 if key != "c" else 60), transform=ax.transAxes, ha="center", va="top", fontsize=FONT_TITLE, fontweight="bold", bbox=bb, zorder=100)
                ax._fixed = (aspect, ax.get_xlim(), ax.get_ylim())
                return ax
            
            def add_tiles(ax, name, alpha):
                if cx is None: return False
                src = {"topo": cx.providers.Esri.WorldTopoMap, "imagery": cx.providers.Esri.WorldImagery, "street": cx.providers.OpenStreetMap.Mapnik, "light": cx.providers.CartoDB.Positron}[name]
                try: cx.add_basemap(ax, crs="EPSG:4326", source=src, alpha=alpha, attribution=False, zorder=1); return True
                except: return False
                
            def fix_axes(ax): ax.set_aspect(ax._fixed[0], adjustable="datalim"); ax.set_xlim(ax._fixed[1]); ax.set_ylim(ax._fixed[2]); ax.apply_aspect()

            OUTER = rect(FIG_LEFT, FIG_BOTTOM, FIG_RIGHT, FIG_TOP)
            PANEL_A = rect(FIG_LEFT + GAP_X, MID_Y + GAP_Y / 2, MID_X - GAP_X, FIG_TOP - GAP_Y)
            PANEL_B = rect(FIG_LEFT + GAP_X, FIG_BOTTOM + GAP_Y, MID_X - GAP_X, MID_Y - GAP_Y / 2)
            PANEL_C = rect(MID_X + GAP_X, FIG_BOTTOM + GAP_Y, FIG_RIGHT - GAP_X, FIG_TOP - GAP_Y)
            used_basemaps = []

            fig.add_artist(Rectangle((OUTER[0], OUTER[1]), OUTER[2], OUTER[3], fill=False, edgecolor=COL_FRAME, linewidth=LW_OUTER, transform=fig.transFigure, clip_on=False))
            fig.add_artist(mlines.Line2D([MID_X, MID_X], [FIG_BOTTOM, FIG_TOP], transform=fig.transFigure, color=COL_FRAME, linewidth=LW_OUTER))

            # --- Panel A ---
            ax_a = make_map(fig, PANEL_A, INNER_A, country_gdf.total_bounds, TITLE_A, PAD_A, FMT_A, NBINS_A, "a")
            has_basemap_a = SHOW_BASEMAP_A and add_tiles(ax_a, BASEMAP_A, BASEMAP_A_ALPHA)
            if has_basemap_a: used_basemaps.append(BASEMAP_A)
            
            if neigh_gdf is not None and len(neigh_gdf):
                if not has_basemap_a: ax_a.set_facecolor(to_rgba(COL_SEA, SEA_ALPHA))
                ax_a.plot(neigh_gdf, facecolor=to_rgba(COL_NEIGH_FILL, 0.15 if has_basemap_a else NEIGH_ALPHA), edgecolor=COL_NEIGH_EDGE, linewidth=0.4, zorder=0.5)

            country_gdf.plot(ax=ax_a, facecolor=to_rgba(COL_COUNTRY_FILL, 0.45 if has_basemap_a else 1.0), edgecolor=COL_COUNTRY_LINE, linewidth=0.35, zorder=1)
            country_gdf.boundary.plot(ax=ax_a, color=COL_COUNTRY_EDGE, linewidth=0.8, zorder=2)
            bx0, by0, bx1, by1 = b_bounds
            ax_a.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, edgecolor=COL_LOCATOR, linewidth=1.8, zorder=20))
            fix_axes(ax_a)

            # --- Panel B ---
            ax_b = make_map(fig, PANEL_B, INNER_B, b_bounds, TITLE_B, PAD_B, FMT_B, NBINS_B, "b")
            fill_b = to_rgba(COL_DISTRICT_FILL, DISTRICT_FILL_ALPHA if SHOW_BASEMAP_B else 1.0)
            if SHOW_BASEMAP_B and add_tiles(ax_b, BASEMAP_B, BASEMAP_B_ALPHA): used_basemaps.append(BASEMAP_B)
            context_gdf.plot(ax=ax_b, facecolor=fill_b, edgecolor=COL_DISTRICT_EDGE, linewidth=0.6, zorder=2)
            if upazilas_gdf is not None: upazilas_gdf.boundary.plot(ax=ax_b, color=COL_UPAZILA_LINE, linewidth=0.5, zorder=3)
            focal.boundary.plot(ax=ax_b, color=COL_FOCAL_EDGE, linewidth=1.1, zorder=4)
            aoi_gdf.boundary.plot(ax=ax_b, color=COL_AOI, linewidth=AOI_LW_B, zorder=20)
            fix_axes(ax_b)

            # --- Panel C ---
            ax_c = make_map(fig, PANEL_C, INNER_C, aoi_union.bounds, TITLE_C, PAD_C, FMT_C, NBINS_C, "c")
            if SHOW_BASEMAP_C and add_tiles(ax_c, BASEMAP_C, BASEMAP_ALPHA): used_basemaps.append(BASEMAP_C)
            else: aoi_gdf.plot(ax=ax_c, facecolor="#F4F4F4", edgecolor="none", zorder=1)

            if PANEL_C_LAYER == "DEM" and dem_arr is not None:
                ab = aoi_union.bounds
                im = ax_c.imshow(dem_arr, cmap=DEM_CMAP, vmin=vmin, vmax=vmax, alpha=DEM_ALPHA, zorder=2, extent=[ab[0], ab[2], ab[1], ab[3]], aspect=ax_c._fixed[0])
                cax = ax_c.inset_axes(COLORBAR_C_POS)
                cb = fig.colorbar(im, cax=cax, orientation="horizontal")
                cb.set_label("Elevation (m)", fontsize=FONT_MAIN, fontweight="bold")
                cb.ax.tick_params(labelsize=FONT_TICK)

            aoi_gdf.boundary.plot(ax=ax_c, color=COL_AOI, linewidth=AOI_LW_C, zorder=20)
            fix_axes(ax_c)

            pb = ax_b.get_position()
            fig.add_artist(ConnectionPatch(xyA=((bx0 + bx1) / 2, by0), coordsA=ax_a.transData, xyB=((pb.x0 + pb.x1) / 2, PANEL_B[1] + PANEL_B[3]), coordsB="figure fraction", arrowstyle="-|>", mutation_scale=13, color=COL_LOCATOR, linewidth=1.1, zorder=200))

            fig.text(FIG_LEFT + 0.01, FIG_BOTTOM - 0.02, textwrap.fill(CAPTION, 95), ha="left", va="top", fontsize=FONT_MAIN + 1)
            
            # --- MARKETING WATERMARK ---
            fig.text(FIG_RIGHT, FIG_BOTTOM - 0.04, "Figure orientation credit @imtiajiqbalmahfuj", ha="right", va="bottom", fontsize=8, color="#888888")

        # Display in Streamlit
        st.pyplot(fig)
