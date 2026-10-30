"""Draws the SeptiSense wiring diagram and the electrode/sensor placement picture."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Ellipse, Rectangle, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
INK = '#16202a'
COL = {'3V3': '#d32f2f', 'GND': '#212121', 'SDA': '#1e5bd8', 'SCL': '#e0a100',
       'ECG': '#2e9b47', 'LO+': '#8e44ad', 'LO-': '#c2185b', 'SDN': '#ef6c00'}


def board(ax, x, y, w, h, title, sub, pins, side, fc):
    """pins: list of names top->bottom; side 'L' or 'R' = where the pin stubs are. Returns {name: (x,y)}."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.18', fc=fc, ec=INK, lw=1.6))
    ax.text(x + w / 2, y + h + 0.32, title, ha='center', va='bottom', fontsize=12.5, weight='bold', color=INK)
    if sub:
        ax.text(x + w / 2, y - 0.25, sub, ha='center', va='top', fontsize=8.6, color='#4a5560')
    out = {}
    n = len(pins)
    for k, p in enumerate(pins):
        py = y + h - (k + 0.5) * h / n
        px = x if side == 'L' else x + w
        s = -1 if side == 'L' else 1
        ax.plot([px, px + s * 0.28], [py, py], color=INK, lw=1.4)
        ax.add_patch(Circle((px + s * 0.28, py), 0.06, fc='white', ec=INK, lw=1.2, zorder=5))
        ax.text(px - s * 0.12, py, p, ha='right' if side == 'R' else 'left', va='center', fontsize=9.6,
                family='DejaVu Sans Mono', color=INK)
        out[p] = (px + s * 0.28, py)
    return out


def wire(ax, a, b, color, via_x=None, lw=2.6):
    (x1, y1), (x2, y2) = a, b
    xm = via_x if via_x is not None else (x1 + x2) / 2
    ax.plot([x1, xm, xm, x2], [y1, y1, y2, y2], color=color, lw=lw, solid_capstyle='round',
            solid_joinstyle='round', zorder=3)
    for (x, y) in (a, b):
        ax.add_patch(Circle((x, y), 0.07, fc=color, ec=color, zorder=6))


def wiring(path, s3=False):
    fig, ax = plt.subplots(figsize=(15, 9.6))
    ax.set_xlim(0, 30); ax.set_ylim(0, 19.2); ax.axis('off')
    tag = 'ESP32-S3 DevKitC' if s3 else 'ESP32 DevKit V1'
    P = {'SDA': 'GPIO8', 'SCL': 'GPIO9', 'ECG': 'GPIO4', 'LO+': 'GPIO5', 'LO-': 'GPIO6'} if s3 else \
        {'SDA': 'GPIO21', 'SCL': 'GPIO22', 'ECG': 'GPIO34', 'LO+': 'GPIO35', 'LO-': 'GPIO32'}
    ax.text(0.3, 18.75, f'SeptiSense monitoring node - wiring ({tag})', fontsize=16, weight='bold', color=INK)
    ax.text(0.3, 18.1, 'All modules run from the ESP32 3V3 pin. The three I2C modules share SDA and SCL (one bus, 100 kHz). '
            'Power the ESP32 from the laptop USB.', fontsize=10, color='#4a5560')

    esp = board(ax, 12.2, 5.0, 4.6, 9.6, tag, '', [], 'L', '#e8eef5')
    ax.add_patch(Rectangle((13.9, 4.45), 1.2, 0.55, fc='#9aa3ad', ec=INK, lw=1.2))
    ax.text(14.5, 4.72, 'USB', ha='center', va='center', fontsize=8, color='white', weight='bold')
    ax.text(14.5, 4.0, 'USB cable to the laptop\n(power + programming)', ha='center', va='top', fontsize=8.6, color='#4a5560')
    # pins: left = ECG side, right = I2C/power side
    def pin(name, side, y):
        x = 12.2 if side == 'L' else 16.8
        s_ = -1 if side == 'L' else 1
        ax.plot([x, x + s_ * 0.28], [y, y], color=INK, lw=1.4)
        ax.add_patch(Circle((x + s_ * 0.28, y), 0.06, fc='white', ec=INK, lw=1.2, zorder=5))
        ax.text(x - s_ * 0.12, y, name, ha='left' if side == 'L' else 'right', va='center', fontsize=9.6,
                family='DejaVu Sans Mono', color=INK)
        return (x + s_ * 0.28, y)
    e = {}
    pitch = 4.6 / 6
    for k, nm in enumerate(['GND', P['ECG'], P['LO+'], P['LO-']]):     # same heights as the AD8232 pins
        e['L_' + nm] = pin(nm, 'L', 10.0 + 4.6 - (k + 1.5) * pitch)
    for k, nm in enumerate(['3V3', 'GND', P['SDA'], P['SCL']]):        # between the MAX30102 and MLX90614 rows
        e['R_' + nm] = pin(nm, 'R', 13.0 - k * 0.5)

    ad = board(ax, 1.4, 10.0, 4.4, 4.6, 'AD8232 ECG', '',
               ['3.3V', 'GND', 'OUTPUT', 'LO+', 'LO-', 'SDN'], 'R', '#e9f6ec')
    mx = board(ax, 22.6, 13.4, 4.0, 2.6, 'MAX30102', 'pulse + SpO2  |  I2C 0x57',
               ['VIN', 'GND', 'SDA', 'SCL'], 'L', '#fdecef')
    ml = board(ax, 22.6, 8.6, 4.0, 2.6, 'MLX90614 (GY-906)', 'IR temperature  |  I2C 0x5A',
               ['VIN', 'GND', 'SDA', 'SCL'], 'L', '#fff5e0')
    ol = board(ax, 22.6, 3.8, 4.0, 2.6, 'OLED 0.96" SSD1306', 'display  |  I2C 0x3C',
               ['VCC', 'GND', 'SDA', 'SCL'], 'L', '#eceaf9')

    # vertical rails between ESP and modules
    rail = {'3V3': 18.2, 'GND': 19.0, 'SDA': 19.8, 'SCL': 20.6}
    src = {'3V3': e['R_3V3'], 'GND': e['R_GND'], 'SDA': e['R_' + P['SDA']], 'SCL': e['R_' + P['SCL']]}
    tops = {k: 16.6 for k in rail}
    for name, x in rail.items():
        x0, y0 = src[name]
        ax.plot([x0, x], [y0, y0], color=COL[name], lw=2.6, zorder=3)
        ax.add_patch(Circle((x0, y0), 0.07, fc=COL[name], ec=COL[name], zorder=6))
        ax.add_patch(Circle((x, y0), 0.09, fc=COL[name], ec=COL[name], zorder=6))
        ys = [y0]
        for mod, pinname in ((mx, 'VIN' if name == '3V3' else name), (ml, 'VIN' if name == '3V3' else name),
                             (ol, 'VCC' if name == '3V3' else name)):
            px, py = mod[pinname]
            ys.append(py)
            ax.plot([x, px], [py, py], color=COL[name], lw=2.6, zorder=3)
            ax.add_patch(Circle((x, py), 0.09, fc=COL[name], ec=COL[name], zorder=6))
            ax.add_patch(Circle((px, py), 0.07, fc=COL[name], ec=COL[name], zorder=6))
        top = max(ys) if name != '3V3' else 16.9
        ax.plot([x, x], [min(ys), top], color=COL[name], lw=2.6, zorder=3)
        ax.text(x, min(ys) - 0.45, name, ha='center', va='top', fontsize=8.4, color=COL[name], weight='bold')

    # AD8232 -> ESP
    a = ad
    x3, y3 = a['3.3V']
    ax.plot([x3, 9.0, 9.0, rail['3V3'], rail['3V3']], [y3, y3, 16.9, 16.9, 16.0], color=COL['3V3'], lw=2.6, zorder=3)
    ax.add_patch(Circle((x3, y3), 0.07, fc=COL['3V3'], ec=COL['3V3'], zorder=6))
    ax.text(10.6, 17.15, 'AD8232 3.3V joins the 3V3 line', fontsize=8.4, color=COL['3V3'])
    wire(ax, a['GND'], e['L_GND'], COL['GND'])
    wire(ax, a['OUTPUT'], e['L_' + P['ECG']], COL['ECG'])
    wire(ax, a['LO+'], e['L_' + P['LO+']], COL['LO+'])
    wire(ax, a['LO-'], e['L_' + P['LO-']], COL['LO-'])
    sx, sy = a['SDN']
    ax.plot([sx, 6.5, 6.5, 8.6, 8.6, 9.0], [sy, sy, 9.7, 9.7, y3, y3], color=COL['SDN'], lw=2.0, ls=(0, (4, 2)), zorder=2)
    ax.text(6.7, 9.45, 'SDN to 3.3V (jumper)\nkeeps the AD8232 on', fontsize=8.4, color=COL['SDN'], va='top')

    # ECG cable + electrodes
    ax.plot([3.6, 3.6], [10.0, 7.5], color='#555', lw=3)
    ax.text(3.75, 9.1, 'ECG cable into the\n3.5 mm jack', ha='left', va='center', fontsize=9, color='#333')
    for k, (lab, c) in enumerate((('RA', '#d32f2f'), ('LA', '#e0a100'), ('RL', '#2e9b47'))):
        x = 2.3 + k * 1.3
        ax.plot([3.6, x], [7.5, 6.6], color='#777', lw=1.6)
        ax.add_patch(Circle((x, 6.25), 0.36, fc=c, ec=INK, lw=1.2))
        ax.text(x, 6.25, lab, ha='center', va='center', fontsize=8.2, color='white', weight='bold')
    ax.text(3.6, 5.5, 'gel stickers on the body\n(see placement picture)', ha='center', va='top', fontsize=8.6, color='#4a5560')

    # legend
    ly = 2.4
    items = [('3V3 power', COL['3V3']), ('GND', COL['GND']), ('I2C SDA', COL['SDA']), ('I2C SCL', COL['SCL']),
             ('ECG analog', COL['ECG']), ('lead-off LO+', COL['LO+']), ('lead-off LO-', COL['LO-'])]
    for k, (t, c) in enumerate(items):
        x = 0.4 + k * 4.15
        ax.plot([x, x + 0.8], [ly, ly], color=c, lw=3)
        ax.text(x + 1.0, ly, t, va='center', fontsize=9.4, color=INK)
    ax.text(0.4, 1.3, 'Analog ECG must use an ADC1 pin (ADC2 stops working while Wi-Fi is on).  '
            'MAX30102 INT pin: leave unconnected.  AD8232 is 3.3 V only - never connect it to 5V/VIN.',
            fontsize=9.2, color='#4a5560')
    fig.savefig(path, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def placement(path):
    fig, ax = plt.subplots(figsize=(11, 8.2))
    ax.set_xlim(0, 22); ax.set_ylim(0, 16.4); ax.set_aspect('equal'); ax.axis('off')
    ax.text(0.3, 15.9, 'Where the sensors go', fontsize=16, weight='bold', color=INK)
    body = '#f2dccb'
    # torso
    ax.add_patch(Circle((6, 13.3), 1.25, fc=body, ec=INK, lw=1.5))
    ax.add_patch(Rectangle((5.45, 11.6), 1.1, 0.6, fc=body, ec=body))
    torso = Polygon([(2.6, 11.7), (9.4, 11.7), (9.0, 4.2), (3.0, 4.2)], closed=True, fc=body, ec=INK, lw=1.5)
    ax.add_patch(torso)
    for xs in ((2.6, 1.3, 1.0), (9.4, 10.7, 11.0)):
        ax.add_patch(Polygon([(xs[0], 11.7), (xs[1], 11.0), (xs[2], 5.4), (xs[2] + (0.9 if xs[0] < 6 else -0.9), 5.4),
                              (xs[0] + (0.0 if xs[0] < 6 else 0.0), 10.0)], closed=True, fc=body, ec=INK, lw=1.4))
    ax.plot([4.0, 5.4], [11.2, 10.95], color='#b08f7a', lw=1.2); ax.plot([8.0, 6.6], [11.2, 10.95], color='#b08f7a', lw=1.2)
    ax.text(6, 3.6, '(front view - the person\'s right is on YOUR left)', ha='center', fontsize=9, color='#4a5560')
    pts = [('RA', 3.9, 10.3, '#d32f2f', 'RA  - right upper chest,\njust below the collarbone'),
           ('LA', 8.2, 6.2, '#e0a100', 'LA  - left lower ribs\n(gives lead II: clearer heartbeats)'),
           ('RL', 3.8, 6.0, '#2e9b47', 'RL  - right lower ribs / abdomen\n(reference electrode)')]
    for lab, x, y, c, txt in pts:
        ax.add_patch(Circle((x, y), 0.42, fc=c, ec=INK, lw=1.3, zorder=5))
        ax.text(x, y, lab, ha='center', va='center', fontsize=8.6, color='white', weight='bold', zorder=6)
    ax.text(12.6, 13.8, 'ECG (AD8232)', fontsize=12.5, weight='bold', color=INK)
    yy = 13.0
    for lab, x, y, c, txt in pts:
        ax.add_patch(Circle((12.9, yy - 0.15), 0.3, fc=c, ec=INK, lw=1.1))
        ax.text(13.5, yy, txt, fontsize=9.6, va='top', color=INK)
        yy -= 1.35
    ax.text(12.6, 8.2, 'Cable colours vary between kits - always go by the\nRA / LA / RL (or R / L / F) printed on each clip.',
            fontsize=9, color='#4a5560', va='top')
    # finger + temp
    ax.text(12.6, 6.6, 'MAX30102 (pulse + SpO₂)', fontsize=12.5, weight='bold', color=INK)
    ax.text(12.6, 6.1, 'Index fingertip resting flat on the sensor window, light pressure.\n'
            'Hold it with a loose band or tape; cover it from bright light.\nKeep the hand still and at heart level.',
            fontsize=9.6, va='top', color=INK)
    ax.text(12.6, 3.6, 'MLX90614 (temperature)', fontsize=12.5, weight='bold', color=INK)
    ax.text(12.6, 3.1, 'Aim at the forehead or inner wrist from 1-2 cm.\nIt reads SKIN temperature (about 33-35 °C), which\n'
            'the firmware converts to an estimated core value.', fontsize=9.6, va='top', color=INK)
    fig.savefig(path, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    wiring(os.path.join(HERE, 'wiring_esp32_devkit_v1.png'))
    wiring(os.path.join(HERE, 'wiring_esp32_s3.png'), s3=True)
    placement(os.path.join(HERE, 'sensor_placement.png'))
    print('done')
