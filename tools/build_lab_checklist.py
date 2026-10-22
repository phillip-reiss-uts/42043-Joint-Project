#!/usr/bin/env python3
"""Build the printable lab checklist (DOCX; convert to PDF with LibreOffice).

    python tools/build_lab_checklist.py
    <libreoffice-kit> convert --input Lab-Checklist-Vision-Guided-Pick.docx \
        --output Lab-Checklist-Vision-Guided-Pick.pdf

Kept as a script so the checklist can be edited without re-laying it out by
hand. The output goes to the workspace root, next to the full notes.
"""

import os

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The checklist lives at the WORKSPACE root, next to the full notes, not
# inside the repo.
ROOT = os.path.dirname(REPO)
OUT = os.path.join(ROOT, 'Lab-Checklist-Vision-Guided-Pick.docx')
ACCENT = RGBColor(0x1F, 0x38, 0x64)
GREY = RGBColor(0x59, 0x59, 0x59)


def build():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.top_margin = sec.bottom_margin = Cm(1.6)
    sec.left_margin = sec.right_margin = Cm(1.8)

    st = doc.styles['Normal']
    st.font.name = 'Calibri'
    st.font.size = Pt(9.5)
    st.paragraph_format.space_after = Pt(3)
    st.paragraph_format.line_spacing = 1.0
    st.element.rPr.rFonts.set(qn('w:eastAsia'), 'Calibri')

    for name, size in (('Heading 1', 13), ('Heading 2', 10.5)):
        h = doc.styles[name]
        h.font.name = 'Calibri'
        h.font.size = Pt(size)
        h.font.bold = True
        h.font.color.rgb = ACCENT
        h.paragraph_format.space_before = Pt(10 if name == 'Heading 1' else 6)
        h.paragraph_format.space_after = Pt(3)
        h.paragraph_format.keep_with_next = True

    def shade(cell, hexcol):
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'), 'clear')
        shd.set(qn('w:fill'), hexcol)
        tcPr.append(shd)

    def cell_text(cell, text, bold=False, size=8.5, color=None):
        cell.text = ''
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(1)
        r = p.add_run(text)
        r.bold = bold
        r.font.size = Pt(size)
        if color is not None:
            r.font.color.rgb = color

    def table(headers, rows, widths):
        t = doc.add_table(rows=1, cols=len(headers))
        t.style = 'Table Grid'
        t.alignment = WD_TABLE_ALIGNMENT.LEFT
        for i, h in enumerate(headers):
            c = t.rows[0].cells[i]
            cell_text(c, h, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
            shade(c, '1F3864')
        for row in rows:
            cells = t.add_row().cells
            for i, val in enumerate(row):
                cell_text(cells[i], val)
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
        # repeat the header if a table ever splits, and keep rows intact
        trPr = t.rows[0]._tr.get_or_add_trPr()
        th = OxmlElement('w:tblHeader')
        th.set(qn('w:val'), 'true')
        trPr.append(th)
        for row in t.rows:
            row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
        return t

    def bullet(text):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.left_indent = Cm(0.5)
        p.add_run(text)

    def check(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.left_indent = Cm(0.55)
        p.paragraph_format.first_line_indent = Cm(-0.55)
        r = p.add_run('[  ]  ')
        r.bold = True
        p.add_run(text)

    def labelled(label, text):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(label + ' ')
        r.bold = True
        p.add_run(text)

    def note(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.left_indent = Cm(0.2)
        r = p.add_run(text)
        r.italic = True
        r.font.size = Pt(9)
        r.font.color.rgb = GREY

    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fp.add_run('Lab checklist  |  Vision-guided pick  |  page ')
    run.font.size = Pt(8)
    run.font.color.rgb = GREY
    r = fp.add_run()
    f1 = OxmlElement('w:fldChar')
    f1.set(qn('w:fldCharType'), 'begin')
    it = OxmlElement('w:instrText')
    it.set(qn('xml:space'), 'preserve')
    it.text = 'PAGE'
    f2 = OxmlElement('w:fldChar')
    f2.set(qn('w:fldCharType'), 'end')
    r._r.append(f1)
    r._r.append(it)
    r._r.append(f2)
    r.font.size = Pt(8)
    r.font.color.rgb = GREY

    # title
    t = doc.add_paragraph()
    t.paragraph_format.space_after = Pt(0)
    r = t.add_run('Lab Checklist — Vision-Guided Pick')
    r.bold = True
    r.font.size = Pt(18)
    r.font.color.rgb = ACCENT

    s = doc.add_paragraph()
    s.paragraph_format.space_after = Pt(2)
    r = s.add_run('UR3e + Robotiq 2F-85 + RealSense D435i')
    r.font.size = Pt(10.5)
    r.font.color.rgb = GREY

    meta = doc.add_paragraph()
    meta.paragraph_format.space_after = Pt(8)
    r = meta.add_run('Session: ______________________     Operator: '
                     '______________________     Robot IP: _______________')
    r.font.size = Pt(9)
    r.font.color.rgb = GREY

    note('Order matters. Tests 1–3 catch the errors that would otherwise cost '
         'the whole session: kinematics, frame convention, and the servoj call '
         'itself. Do not skip to the pick. Every number here is a simulation '
         'expectation, not a hardware guarantee — record what you actually see.')

    doc.add_heading('Before the lab — offline, 20 min', level=1)
    check('All eight suites pass on the laptop: test_layering, test_task_flow, '
          'test_dynamics, test_ur_kinematics, test_ur_backend, '
          'test_fixed_camera, test_calibration_camera, test_hardware.py --sim.')
    check('ur_rtde, pymodbus and pyrealsense2 (with librealsense) installed on '
          'the robot laptop.')
    check('UR_IP set in src/config.py.')
    check('Camera mode decided in src/config.py: CAMERA_MODE = \'wrist\' or '
          '\'fixed\'. If fixed, run tools/calibrate_camera.py (--touch or '
          '--aruco) and paste FIXED_CAM_POS / TARGET / UP in before the '
          'session (Appendix).')
    check('Table measured: centre within a few mm of TABLE_CENTRE = '
          '(0.275, 0.035) from the base; top at TABLE_HEIGHT = 0.02 m.')
    check('Real cube measured against OBJECT_SIZE — half-extents of 0.03 mean '
          'a 60 mm cube, whatever the old notes say.')
    check('Gripper cycle budget decided: STEPS_CLOSE 3000 / STEPS_RELEASE 2000 '
          'are control cycles, so 24 s / 16 s at 125 Hz. Lower them now if '
          'that is too slow (it also changes the simulation).')

    doc.add_heading('0.  Safety and connection  (5 min)', level=1)
    labelled('Run:', 'Pendant in Remote, external-control program loaded, '
                     'brakes released, speed slider ~10%. Call connect() and '
                     'read joint_positions().')
    labelled('Expect:', 'Six sensible joint angles; no exception from connect().')
    labelled('If it fails:', 'Wrong IP or subnet, robot not in Remote, or the '
                             'program not running. A pymodbus TypeError is the '
                             'slave=/unit= kwarg difference — the code tries '
                             'both.')

    doc.add_heading('1.  FK and mount versus the controller  (15 min) — the '
                    'most important test', level=1)
    labelled('Run:', 'No motion. Hand-guide the arm to 5 poses. At each, record '
                     'q and getActualTCPPose(), then compare with '
                     'URKinematics.pose(q), which returns the gripper base.')
    labelled('Expect:', '|Δp| < 1 mm and < 0.5° after accounting for which '
                        'frame the pendant calls "TCP" (tool0/flange vs '
                        'gripper base — they differ by TOOL_MOUNT, about '
                        '10.8 mm and 90°).')
    labelled('If it fails:', '')
    bullet('a constant offset only → TOOL_MOUNT_POS/QUAT, or the pendant TCP '
           'is the flange. Compare wrist_3 to tool0 first.')
    bullet('a constant rotation (sign flips, 90° / 180°) → frame convention. '
           'Check _root_transform and whether the model base is rotated.')
    bullet('error grows with configuration → link data is wrong (payload, '
           'merged gripper mass). Re-derive models/ur3e/arm_dynamics.json with '
           'the real gripper fitted.')
    bullet('correct at some poses only → joint order or zero offsets; '
           'UR_JOINT_NAMES must match getActualQ().')

    table(['Pose', 'TCP x, y, z (mm)', 'FK x, y, z (mm)', 'Δp (mm)',
           'Δ angle (°)'],
          [['1', '', '', '', ''], ['2', '', '', '', ''],
           ['3', '', '', '', ''], ['4', '', '', '', ''],
           ['5', '', '', '', '']],
          [1.4, 4.2, 4.2, 2.4, 2.4])

    doc.add_heading('2.  First motion — one joint, 2°  (10 min)', level=1)
    labelled('Run:', 'arm.move with rate=FEED on a single joint, +2°. Time a '
                     'few backend.step() calls.')
    labelled('Expect:', 'Smooth motion in the right direction; each step ≈ '
                        '8 ms; joint_positions() tracks the command.')
    labelled('If it fails:', 'A TypeError from servoj means the keyword form '
                             'servoj(q, time=…, lookahead_time=…, gain=…) is '
                             'not accepted — switch to positional arguments. '
                             'Motion without tracking means the controller is '
                             'not in servo mode, or the program stopped.')

    doc.add_heading('3.  Cartesian sense and repeatability  (15 min)', level=1)
    labelled('Run:', 'From a pose above the table, command ±20 mm in x, then y, '
                     'then z. Then return to one pose five times.')
    labelled('Expect:', '+x away from the base, +z up, +y left (UR base '
                        'convention). Repeatability < 2 mm.')
    labelled('If it fails:', 'A mirrored axis is a frame problem, not a gain '
                             'problem — do not retune around it. Right axes but '
                             'wrong magnitude points at TOOL_MOUNT.')

    doc.add_heading('4.  Gripper registers and thresholds  (15 min)', level=1)
    labelled('Run:', 'Activate the 2F-85, then read gPO and gCU for four '
                     'states: fully open, closed on air, closed on the cube, '
                     'closed on a soft object.')
    labelled('Expect:', 'Open near 0; closed on air near 230–255; the cube '
                        'clearly between. grasp_state() says held when q < 0.60 '
                        '(gPO < 191) and force > 1.0 N (gCU > 51).')
    labelled('If it fails:', 'Wrong unit id, wrong register map, or activation '
                             'incomplete (check gACT). Numbers in range but the '
                             'wrong verdict means the Q_SCALE / F_SCALE mapping '
                             'needs calibrating — the force mapping is a '
                             'placeholder, not newtons.')

    table(['Gripper state', 'gPO', 'gCU', 'grasp_state verdict'],
          [['fully open', '', '', ''], ['closed on air', '', '', ''],
           ['closed on cube', '', '', ''],
           ['closed on soft object', '', '', '']],
          [4.6, 2.6, 2.6, 5.0])

    doc.add_heading('5.  Camera  (20 min)', level=1)
    labelled('Run:', 'Set CAMERA_MODE in src/config.py first. If fixed, the '
                     'extrinsics should already be calibrated (Appendix) — '
                     'this test then verifies them. intrinsics_report(); point '
                     'at the empty table; place the cube at three known spots '
                     'and compare the detected centre with a tape measure.')
    labelled('Expect:', 'Depth in millimetres. Deprojected table points within '
                        '±3 mm of the measured TABLE_HEIGHT. Detection within '
                        '3–5 mm at the centre, worse at the corners. With a '
                        'fixed camera the view does not follow the arm — move '
                        'the arm and the estimate must not change.')
    labelled('If it fails:', 'No points → depth units or the table filter '
                             '(SURFACE_BAND = 5 mm). Wrong place in wrist mode '
                             '→ CAM_MOUNT extrinsics; in fixed mode → re-run '
                             'tools/calibrate_camera.py and check the residual. '
                             'No detection → outside the view, or rejected by '
                             '_span_ok’s size band.')

    table(['Position', 'Truth x, y (mm)', 'Detected x, y (mm)', 'Error (mm)'],
          [['centre', '', '', ''], ['near corner', '', '', ''],
           ['far corner', '', '', '']],
          [3.4, 4.4, 4.4, 3.0])

    doc.add_heading('6.  Hover servo — no descent  (15 min)', level=1)
    labelled('Run:', 'TRACK mode, or call the servo at SERVO_STANDOFF = 0.20 m '
                     'with max_iters bounded.')
    labelled('Expect:', 'Converges and holds with < 10 mm error and no limit '
                        'cycle.')
    labelled('If it fails:', 'Oscillation → controller lookahead/gain, or '
                             'SERVO_V_TRANSIT too high. Drift → the servo’s dt '
                             'or time source (getTimestamp). Wrong target → '
                             'back to Test 1.')

    doc.add_heading('7.  Guarded pick, five attempts  (20 min)', level=1)
    labelled('Run:', 'One press of p. The robot should approach, close, verify, '
                     'lift and carry to home — and NOT release.')
    labelled('Expect on day one:', '3 of 5 held; grasp gap < 10 mm; the cube '
                                   'still in the jaws after the carry.')
    labelled('If it fails:', 'Jaws hit the cube on the way down → lower '
                             'SERVO_V_DESCENT. Large gap → Test 1 or Test 5 '
                             'error, not the servo. "Empty" with a small gap → '
                             'Test 4 thresholds, and check the cube height '
                             'against TIP_OFFSET. With a fixed camera, check '
                             'the arm is not hiding the cube before the close.')

    table(['Attempt', 'Gap (mm)', 'Held?', 'Notes'],
          [['1', '', '', ''], ['2', '', '', ''], ['3', '', '', ''],
           ['4', '', '', ''], ['5', '', '', '']],
          [2.0, 2.4, 2.0, 9.0])

    doc.add_heading('8.  Put-down  (10 min)', level=1)
    labelled('Run:', 'Second press of p. Measure where the cube lands and '
                     'whether the pads press into the table.')
    labelled('Expect:', 'Centre within ~10 mm of the tool xy; resting exactly '
                        'on the surface (the simulation lands at 0.0499 m for '
                        'a 60 mm cube).')
    labelled('If it fails:', 'It sinks → TIP_OFFSET or OBJECT_HEIGHT wrong for '
                             'the real cube. Off-centre → Test 1.')

    doc.add_heading('9.  Optional, only if time allows', level=1)
    p = doc.add_paragraph()
    p.add_run('Pause/resume mid-move, occlusion (hand over the cube), target '
              'moved during the descent, search path. All are exercised in '
              'simulation, none on hardware.')

    brk = doc.add_paragraph()
    brk.add_run().add_break(WD_BREAK.PAGE)

    doc.add_heading('Failure triage', level=1)
    table(['Symptom', 'First thing to check'],
          [['TypeError on servoj', 'keyword vs positional signature'],
           ['arm does not move, no error', 'Remote mode / program running'],
           ['motion mirrored', 'base-frame convention, not gains'],
           ['FK off by a constant', 'TOOL_MOUNT, or pendant TCP selection'],
           ['FK off by pose', 'link data / payload'],
           ['gripper silent', 'Modbus unit id, gACT'],
           ['grasp always "empty"', 'gPO thresholds from Test 4'],
           ['grasp "held" on air', 'F_SCALE too high'],
           ['camera points in the wrong plane', 'depth units, camera mount'],
           ['fixed camera: bias grows across the table',
            're-run calibrate_camera.py; check residual'],
           ['calibration residual > 10 mm',
            'touches not centred, or TABLE_HEIGHT wrong'],
           ['aruco: no marker detected', 'lighting, focus, dictionary name'],
           ['hover oscillates', 'lookahead/gain, SERVO_V_*'],
           ['works but is glacial',
            'STEPS_CLOSE / RELEASE are cycles, not seconds']],
          [6.2, 11.2])

    doc.add_heading('Numbers to bring back', level=1)
    p = doc.add_paragraph()
    p.add_run('TCP vs FK per pose; repeatability; gPO/gCU for the four gripper '
              'states; detection error at three positions; calibration '
              'residual and the four config numbers; hover error; five grasp '
              'gaps and outcomes; put-down error. These tell us what to change '
              'next.')

    doc.add_heading('Appendix — fixed camera: calibrate and run it', level=1)
    p = doc.add_paragraph()
    p.add_run('If the camera cannot be wrist-mounted, the project runs '
              'eye-to-hand from the same code: the depth pipeline is shared, '
              'only the camera pose differs. Set CAMERA_MODE = \'fixed\' and '
              'give it the pose below.')

    labelled('What to print:', '')
    bullet('one ArUco marker, DICT_4X4_50, printed edge 100 mm. A ready-to-print '
           'PDF and a 300 dpi PNG are in models/aruco/ — the page is 140 mm and '
           'the black square exactly 100 mm. Print at 100% / actual size, then '
           'MEASURE the black square with calipers and pass that value to '
           '--marker-size. A 2% print-scale error is a 2% calibration error.')
    bullet('matte paper glued flat to card or acrylic. Glossy lamination and '
           'tape wrinkles jitter the corners; a curled board cannot be rescued '
           'by maths.')
    bullet('it must be visible from the camera and not tight against an '
           'obstacle: two adjacent corners have to be touchable.')

    labelled('Calibrate (either method, before the session):', '')
    bullet('--touch  python tools/calibrate_camera.py --touch. Place the cube, '
           'centre the CLOSED gripper over it, lower until the pads just touch '
           'the top face, Enter. Six or more positions across the table. No '
           'OpenCV. The camera HEIGHT is not observable this way, so it is '
           'taken from the config value — measure it first.')
    bullet('--aruco  python tools/calibrate_camera.py --aruco --marker-size '
           '0.100. Capture first, open the saved aruco_corners.png, then touch '
           'the corners it labels 0 and 1. solvePnP observes all six degrees of '
           'freedom, so use this if the height is uncertain. Needs '
           'opencv-contrib-python.')
    bullet('Both print FIXED_CAM_POS / TARGET / UP for src/config.py and a '
           'residual. Under ~5 mm is good; over 10 mm means the touches or '
           'TABLE_HEIGHT are wrong, not the maths.')
    bullet('Then the ruler check: cube at three measured spots, compare '
           'detected centres. This is Test 5.')

    labelled('What changes:', '')
    bullet('the estimate no longer moves with the arm — moving the robot cannot '
           'move the target (tests/test_fixed_camera.py pins this).')
    bullet('the arm can occlude the cube on the way down. Keep the '
           'commit-at-standoff and re-measure SERVO_STANDOFF; the occlusion '
           'heuristic (SERVO_OCLUDE_RATIO) was tuned to the jaws clipping the '
           'wrist view and will need adjusting.')
    bullet('search_for_object sweeps and revolves the wrist; with a fixed '
           'camera those motions change nothing. Expect to replace it with a '
           'fixed-view look or an operator prompt.')
    bullet('restrict the pick region to what the camera sees in one shot.')

    labelled('Test first:', 'python tests/test_calibration_camera.py and '
            'tests/test_fixed_camera.py are offline; then '
            'tools/calibrate_camera.py on the cell, then the ruler check.')
    labelled('Mount advice:', 'high and angled down (~45°) so the arm crosses '
            'the view less; a low side view loses the cube during the descent.')

    doc.core_properties.title = 'Lab Checklist - Vision-Guided Pick'
    doc.core_properties.subject = 'UR3e + Robotiq 2F-85 + D435i hardware bring-up'
    doc.save(OUT)
    return OUT


if __name__ == '__main__':
    print('wrote', build())
