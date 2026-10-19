
import json
import os
import struct

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, PointerProperty, StringProperty
from bpy_extras.io_utils import ExportHelper

BANDS = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]

MATERIALS = [
    ('default',  'Demo default',                 [0.20, 0.22, 0.25, 0.27, 0.30, 0.33, 0.35, 0.37], 0.20),
    ('concrete', 'Concrete (smooth)',            [0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.04, 0.05], 0.05),
    ('brick',    'Brick (unglazed)',             [0.03, 0.03, 0.03, 0.04, 0.05, 0.07, 0.08, 0.09], 0.10),
    ('marble',   'Marble / glazed tile',         [0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03], 0.05),
    ('plaster',  'Plasterboard on studs',        [0.29, 0.10, 0.05, 0.04, 0.07, 0.09, 0.10, 0.11], 0.05),
    ('wood',     'Wood panelling',               [0.28, 0.22, 0.17, 0.09, 0.10, 0.11, 0.12, 0.13], 0.10),
    ('glass',    'Glass (window)',               [0.35, 0.25, 0.18, 0.12, 0.07, 0.04, 0.03, 0.03], 0.05),
    ('carpet',   'Carpet on concrete',           [0.02, 0.06, 0.14, 0.37, 0.60, 0.65, 0.68, 0.70], 0.20),
    ('curtain',  'Heavy curtains',               [0.14, 0.35, 0.55, 0.72, 0.70, 0.65, 0.65, 0.65], 0.40),
    ('audience', 'Audience / upholstered seats', [0.60, 0.74, 0.88, 0.96, 0.93, 0.85, 0.80, 0.75], 0.70),
    ('foam',     'Acoustic foam (anechoic)',     [0.90, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99], 0.50),
]
GEOMETRY_TYPES = {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT'}
MARKERS = ('Source', 'Listener')


def apply_preset(self, _context):
    for key, _label, absorption, scattering in MATERIALS:
        if key == self.preset:
            self.absorption = absorption
            self.scattering = scattering


class RIRRoomSettings(bpy.types.PropertyGroup):
    room_name: StringProperty(name='Room name', description='Name in the demo\'s room list (empty: scene name)')
    use_absorption: BoolProperty(name='Absorption', default=True,
                                 description='Write extras.absorption (else the demo uses its default)')
    use_scattering: BoolProperty(name='Scattering', default=True,
                                 description='Write extras.scattering (else the demo uses its default)')
    preset: EnumProperty(name='Preset', items=[(k, label, '') for k, label, _a, _s in MATERIALS], update=apply_preset)
    absorption: FloatVectorProperty(name='Absorption', size=len(BANDS), min=0.0, max=1.0, precision=2, step=1,
                                    default=MATERIALS[0][2], description='Energy absorption per octave band')
    scattering: FloatProperty(name='Scattering', min=0.0, max=1.0, precision=2, step=1, default=MATERIALS[0][3])
    source: PointerProperty(name='Source', type=bpy.types.Object,
                            description='Source start position (empty: an object named "Source")')
    listener: PointerProperty(name='Listener', type=bpy.types.Object,
                              description='Listener start position (empty: an object named "Listener")')
    use_selection: BoolProperty(name='Selected only', default=False,
                                description='Export only selected objects (else every visible one)')
    export_materials: BoolProperty(name='Materials', default=True,
                                   description='Export materials (only used for display in the demo)')
    update_index: BoolProperty(name='Update index.json', default=True,
                               description='Rebuild index.json (the demo\'s room list) in the output folder')
    filepath: StringProperty(name='File', subtype='FILE_PATH', default='//models/room.glb',
                             description='Output .glb for the sidebar\'s Export button')


def marker_object(scene, settings, name):
    obj = getattr(settings, name.lower())
    if obj:
        return obj
    for o in scene.objects:
        if o.name.split('.')[0].lower() == name.lower():
            return o
    return None


def read_glb(path):
    with open(path, 'rb') as f:
        data = f.read()
    magic, _version, _length = struct.unpack_from('<III', data, 0)
    if magic != 0x46546C67:
        raise ValueError(f'{path} is not a GLB')
    json_len, _ = struct.unpack_from('<II', data, 12)
    doc = json.loads(data[20:20 + json_len])
    rest = data[20 + json_len:]
    bin_chunk = b''
    if len(rest) >= 8:
        bin_len, _ = struct.unpack_from('<II', rest, 0)
        bin_chunk = rest[8:8 + bin_len]
    return doc, bin_chunk


def write_glb(path, doc, bin_chunk):
    js = json.dumps(doc, separators=(',', ':')).encode()
    js += b' ' * (-len(js) % 4)
    bin_chunk += b'\0' * (-len(bin_chunk) % 4)
    total = 12 + 8 + len(js) + (8 + len(bin_chunk) if bin_chunk else 0)
    with open(path, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, 2, total))
        f.write(struct.pack('<II', len(js), 0x4E4F534A) + js)
        if bin_chunk:
            f.write(struct.pack('<II', len(bin_chunk), 0x004E4942) + bin_chunk)


def write_index(folder):
    """index.json in folder: every .glb / .gltf there as { file, name } (name = glTF scene name)."""
    rooms = []
    for file in os.listdir(folder):
        if not file.lower().endswith(('.glb', '.gltf')):
            continue
        name = None
        try:
            if file.lower().endswith('.glb'):
                doc, _ = read_glb(os.path.join(folder, file))
            else:
                with open(os.path.join(folder, file), encoding='utf-8') as f:
                    doc = json.load(f)
            name = doc.get('scenes', [{}])[doc.get('scene', 0)].get('name')
        except Exception:
            pass
        rooms.append({'file': file, 'name': name or os.path.splitext(file)[0]})
    rooms.sort(key=lambda r: r['name'].lower())
    with open(os.path.join(folder, 'index.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(rooms, indent=2) + '\n')
    return len(rooms)


def gltf_operator_kwargs(**wanted):
    """Only the options this Blender's glTF exporter knows (names drift between versions)."""
    known = bpy.ops.export_scene.gltf.get_rna_type().properties.keys()
    return {k: v for k, v in wanted.items() if k in known}


def export_room(context, filepath, settings):
    scene = context.scene
    filepath = bpy.path.abspath(filepath)
    if not filepath.lower().endswith('.glb'):
        filepath += '.glb'
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

    markers = {name: marker_object(scene, settings, name) for name in MARKERS}
    marker_set = {o for o in markers.values() if o}
    pool = context.selected_objects if settings.use_selection else context.visible_objects
    objects = [o for o in pool if o.type in GEOMETRY_TYPES and o not in marker_set]
    if not objects:
        raise RuntimeError('nothing to export: no visible' + (' selected' if settings.use_selection else '') +
                           ' mesh objects')

    view_layer = context.view_layer
    old_selected, old_active = list(context.selected_objects), view_layer.objects.active
    if context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    try:
        for o in view_layer.objects:
            o.select_set(o in objects)
        bpy.ops.export_scene.gltf(**gltf_operator_kwargs(
            filepath=filepath, export_format='GLB', use_selection=True, export_apply=True, export_yup=True,
            export_cameras=False, export_lights=False, export_extras=False, export_animations=False,
            export_materials='EXPORT' if settings.export_materials else 'NONE',
        ))
    finally:
        for o in view_layer.objects:
            o.select_set(o in old_selected)
        view_layer.objects.active = old_active

    doc, bin_chunk = read_glb(filepath)
    doc.setdefault('scenes', [{'nodes': []}])
    sc = doc['scenes'][doc.setdefault('scene', 0)]
    sc['name'] = settings.room_name.strip() or scene.name
    extras = {}
    if settings.use_absorption:
        extras['absorption'] = [round(a, 4) for a in settings.absorption]
    if settings.use_scattering:
        extras['scattering'] = round(settings.scattering, 4)
    sc.pop('extras', None)
    if extras:
        sc['extras'] = extras
    nodes = doc.setdefault('nodes', [])
    for name, obj in markers.items():
        if not obj:
            continue
        x, y, z = obj.matrix_world.translation
        nodes.append({'name': name, 'translation': [round(x, 6), round(z, 6), round(-y, 6)]})
        sc.setdefault('nodes', []).append(len(nodes) - 1)
    doc.setdefault('asset', {})['generator'] = 'rir-demo Blender exporter (' + doc['asset'].get('generator', '') + ')'
    write_glb(filepath, doc, bin_chunk)

    msg = f'wrote {filepath}: "{sc["name"]}", {len(objects)} objects'
    missing = [n for n, o in markers.items() if not o]
    if missing:
        msg += ', no ' + ' / '.join(missing) + ' marker (demo picks a default position)'
    if settings.update_index:
        msg += f', index.json: {write_index(os.path.dirname(filepath))} rooms'
    return msg


class RIR_OT_export_room(bpy.types.Operator, ExportHelper):
    """Export the scene as a rir-demo room (.glb with acoustic extras and Source / Listener markers)"""
    bl_idname = 'rir.export_room'
    bl_label = 'Export RIR Room'
    bl_options = {'PRESET'}

    filename_ext = '.glb'
    filter_glob: StringProperty(default='*.glb', options={'HIDDEN'})

    def invoke(self, context, event):
        settings = context.scene.rir_room
        self.filepath = bpy.path.abspath(settings.filepath) if settings.filepath else ''
        if not self.filepath or os.path.basename(self.filepath) == 'room.glb':
            name = settings.room_name.strip() or context.scene.name
            self.filepath = os.path.join(os.path.dirname(self.filepath) or os.getcwd(),
                                         bpy.path.clean_name(name).lower().replace('_', '-') + '.glb')
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def draw(self, context):
        draw_settings(self.layout, context.scene.rir_room, show_file=False)

    def execute(self, context):
        settings = context.scene.rir_room
        try:
            msg = export_room(context, self.filepath, settings)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        settings.filepath = self.filepath
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class RIR_OT_export_room_quick(bpy.types.Operator):
    """Export to the file set in the panel, without a file browser"""
    bl_idname = 'rir.export_room_quick'
    bl_label = 'Export'

    def execute(self, context):
        try:
            msg = export_room(context, context.scene.rir_room.filepath, context.scene.rir_room)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class RIR_OT_add_marker(bpy.types.Operator):
    """Add an empty at the 3D cursor and use it as this marker"""
    bl_idname = 'rir.add_marker'
    bl_label = 'Add Marker'
    bl_options = {'REGISTER', 'UNDO'}

    marker: EnumProperty(items=[(m, m, '') for m in MARKERS])

    def execute(self, context):
        empty = bpy.data.objects.new(self.marker, None)
        empty.empty_display_type = 'SPHERE'
        empty.empty_display_size = 0.15
        empty.location = context.scene.cursor.location
        context.collection.objects.link(empty)
        setattr(context.scene.rir_room, self.marker.lower(), empty)
        return {'FINISHED'}


def draw_settings(layout, settings, show_file=True):
    layout.use_property_split = True
    layout.use_property_decorate = False
    layout.prop(settings, 'room_name')

    box = layout.box()
    box.label(text='Room material (scene extras)')
    box.prop(settings, 'preset')
    row = box.row(heading='Absorption')
    row.prop(settings, 'use_absorption', text='')
    col = box.column(align=True)
    col.active = settings.use_absorption
    for b, hz in enumerate(BANDS):
        col.prop(settings, 'absorption', index=b, text=f'{hz} Hz' if hz < 1000 else f'{hz // 1000} kHz')
    row = box.row(heading='Scattering')
    row.prop(settings, 'use_scattering', text='')
    sub = row.row()
    sub.active = settings.use_scattering
    sub.prop(settings, 'scattering', text='')

    box = layout.box()
    box.label(text='Start positions')
    for m in MARKERS:
        row = box.row(align=True)
        row.prop(settings, m.lower())
        row.operator('rir.add_marker', text='', icon='ADD').marker = m

    box = layout.box()
    box.label(text='Export')
    box.prop(settings, 'use_selection')
    box.prop(settings, 'export_materials')
    box.prop(settings, 'update_index')
    if show_file:
        box.prop(settings, 'filepath')


class RIR_PT_room(bpy.types.Panel):
    bl_label = 'RIR Room'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'RIR'

    def draw(self, context):
        draw_settings(self.layout, context.scene.rir_room)
        row = self.layout.row(align=True)
        row.scale_y = 1.4
        row.operator('rir.export_room_quick', icon='EXPORT')
        row.operator('rir.export_room', text='', icon='FILEBROWSER')


def menu_export(self, _context):
    self.layout.operator(RIR_OT_export_room.bl_idname, text='RIR Room (.glb)')


classes = (RIRRoomSettings, RIR_OT_export_room, RIR_OT_export_room_quick, RIR_OT_add_marker, RIR_PT_room)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.rir_room = PointerProperty(type=RIRRoomSettings)
    bpy.types.TOPBAR_MT_file_export.append(menu_export)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_export)
    del bpy.types.Scene.rir_room
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
