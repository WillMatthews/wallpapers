/*
 * flightsim — an early-90s style flight-sim screensaver.
 *
 * Endless procedural flat-shaded terrain with forests, low-poly cumulus that
 * shade the ground, distance haze, a gradient sky, and an autopilot that banks through valleys and hops ridges, with an optional
 * green vector HUD. Plain Xlib + GLX + fixed-function GL, like the
 * xscreensaver hacks, so it runs the same way: --window, or --root (a
 * full-size child of the root window, which is what the swaylock-plugin /
 * Xwayland chain needs).
 *
 * Spanning: run one instance per monitor with the same --seed, --epoch and
 * --layout (every output's logical rect), and --output = this one's index.
 * They all fly the same deterministic fixed-step sim off the shared wall-clock
 * epoch, through one virtual camera centred on the middle monitor; each renders
 * just its own slice of that camera's view, so the scene is continuous.
 *
 * Keys (window mode): q / Escape quit.
 */
#define _DEFAULT_SOURCE
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <GL/gl.h>
#include <GL/glx.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEG (180.0f / (float)M_PI)

/* ---- options ---- */
static int opt_root = 0, opt_hud = 1;
static float opt_speed = 1.0f, opt_fov = 60.0f;
static const char *opt_time = "random";
static uint32_t opt_seed = 0;
static double opt_epoch = 0;          /* CLOCK_REALTIME seconds the sim starts at */
static int opt_fov_set = 0;
static int opt_trees = 1, opt_clouds = 1;

/* ---- spanning ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];       /* x, y, w, h of each output (logical) */
static int span_n = 0, span_me = -1;

/* ---- terrain ---- */
#define CELL 24.0f
#define HALF 170                 /* cells drawn each side of the camera */
#define RING 512                 /* cache ring size (> 2*HALF+1) */
#define FAR  (HALF * CELL)

static float seed_x, seed_z;

static uint32_t hash2(int32_t x, int32_t z, uint32_t salt) {
    uint32_t h = (uint32_t)x * 0x8da6b343u ^ (uint32_t)z * 0xd8163841u ^ salt * 0xcb1ab31fu;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return h;
}
static float rand01(int32_t x, int32_t z, uint32_t salt) {
    return (hash2(x, z, salt) & 0xffffff) / (float)0x1000000;
}
static float smooth(float t) { return t * t * (3 - 2 * t); }

static float vnoise(float x, float z, uint32_t salt) {
    float fx = floorf(x), fz = floorf(z);
    int32_t ix = (int32_t)fx, iz = (int32_t)fz;
    float tx = smooth(x - fx), tz = smooth(z - fz);
    float a = rand01(ix, iz, salt), b = rand01(ix + 1, iz, salt);
    float c = rand01(ix, iz + 1, salt), d = rand01(ix + 1, iz + 1, salt);
    return (a + (b - a) * tx) + ((c + (d - c) * tx) - (a + (b - a) * tx)) * tz;
}
static float fbm(float x, float z, int oct, uint32_t salt) {
    float s = 0, amp = 0.5f, norm = 0;
    for (int i = 0; i < oct; i++) {
        s += vnoise(x, z, salt + i) * amp;
        norm += amp; amp *= 0.5f; x *= 2.03f; z *= 2.03f;
    }
    return s / norm;
}
static float ridged(float x, float z, int oct, uint32_t salt) {
    float s = 0, amp = 0.5f, norm = 0;
    for (int i = 0; i < oct; i++) {
        float n = 1 - fabsf(vnoise(x, z, salt + i) * 2 - 1);
        s += n * n * amp;
        norm += amp; amp *= 0.5f; x *= 2.07f; z *= 2.07f;
    }
    return s / norm;
}
static float smoothstep(float a, float b, float x) {
    float t = (x - a) / (b - a);
    return smooth(t < 0 ? 0 : t > 1 ? 1 : t);
}

/* Raw height; negative = under the sea. */
static float terrain(float x, float z) {
    x += seed_x; z += seed_z;
    float base = fbm(x / 900, z / 900, 5, 11) * 260;
    float mask = smoothstep(0.50f, 0.78f, fbm(x / 3200, z / 3200, 3, 23));
    float mtn = powf(ridged(x / 750, z / 750, 5, 37), 1.4f) * 950 * mask;
    return base + mtn - 95;
}

#define MAXT 3                   /* trees per cell */
typedef struct {
    int32_t gx, gz;
    float v[6][6];               /* C3F_V3F x 6 */
    float lit;                   /* direct sunlight left after cloud shadow */
    int nt;
    float tr[MAXT][7];           /* x, y, z, height, radius, conifer, shade */
} Cell;
static Cell *cache;
static float *verts;

typedef struct {
    float sky_zen[3], sky_hor[3], sun[3], sun_dir[3], light[3], ambient;
} Palette;
static Palette pal;

static void land_color(float h, float slope, float out[3]) {
    static const float sand[3] = {.76f, .70f, .48f}, grass[3] = {.33f, .58f, .22f},
        lush[3] = {.22f, .45f, .17f}, forest[3] = {.16f, .34f, .14f},
        rock[3] = {.46f, .41f, .35f}, snow[3] = {.95f, .95f, .98f};
    const float *c;
    if (h < 7) c = sand;
    else if (slope > 0.5f && h > 60) c = rock;
    else if (h < 110) c = grass;
    else if (h < 230) c = lush;
    else if (h < 430) c = forest;
    else if (h < 720 || slope > 0.42f) c = rock;
    else c = snow;
    memcpy(out, c, sizeof(float) * 3);
}

/* ---- clouds ---- */
/* A sparse grid of cumulus: each CG cell may hold one cloud, a clump of
 * flat-bottomed puffs. Pure function of the cell, so shadows and drawing
 * (and every spanning instance) agree without storing anything. */
#define CG 520.0f
#define CLOUD_BASE 780.0f
#define CLOUD_SPREAD 120.0f
#define PUFFS 6
typedef struct { float x, z, base; int n; float p[PUFFS][4]; /* dx, dz, r, height scale */ } Cloud;
static int32_t cloud_off_x, cloud_off_z;

static int get_cloud(int32_t i, int32_t j, Cloud *c) {
    int32_t ii = i + cloud_off_x, jj = j + cloud_off_z;
    float cover = 0.12f + 0.6f * fbm(ii * 0.21f, jj * 0.21f, 2, 211);
    if (rand01(ii, jj, 202) >= cover) return 0;
    c->x = (i + 0.2f + 0.6f * rand01(ii, jj, 203)) * CG;
    c->z = (j + 0.2f + 0.6f * rand01(ii, jj, 204)) * CG;
    c->base = CLOUD_BASE + CLOUD_SPREAD * rand01(ii, jj, 205);
    float r0 = 80 + 110 * rand01(ii, jj, 206);
    c->n = 3 + (int)(4 * rand01(ii, jj, 207));
    if (c->n > PUFFS) c->n = PUFFS;
    c->p[0][0] = c->p[0][1] = 0; c->p[0][2] = r0; c->p[0][3] = 0.7f + 0.2f * rand01(ii, jj, 212);
    for (int k = 1; k < c->n; k++) {
        float a = 2 * (float)M_PI * rand01(ii * 8 + k, jj, 208), d = r0 * (0.45f + 0.5f * rand01(ii * 8 + k, jj, 209));
        c->p[k][0] = cosf(a) * d; c->p[k][1] = sinf(a) * d;
        c->p[k][2] = r0 * (0.4f + 0.35f * rand01(ii * 8 + k, jj, 210));
        c->p[k][3] = 0.55f + 0.3f * rand01(ii * 8 + k, jj, 213);
    }
    return 1;
}

/* Fraction of direct sun reaching (x, y, z) past the clouds. */
static float sun_lit(float x, float y, float z) {
    const float *sd = pal.sun_dir;
    if (!opt_clouds || sd[1] < 0.02f) return 1;
    float mid = CLOUD_BASE + CLOUD_SPREAD / 2 + 40, t = (mid - y) / sd[1];
    if (t < 0) return 1;
    float qx = x + sd[0] * t, qz = z + sd[2] * t, hz = sqrtf(sd[0] * sd[0] + sd[2] * sd[2]);
    /* search wide enough for the base-height spread and the largest cloud */
    int n = (int)ceilf(((CLOUD_SPREAD / 2 + 40) / sd[1] * hz + 670) / CG);
    int32_t ci = (int32_t)floorf(qx / CG), cj = (int32_t)floorf(qz / CG);
    float cover = 0;
    for (int dj = -n; dj <= n; dj++)
        for (int di = -n; di <= n; di++) {
            Cloud c;
            if (!get_cloud(ci + di, cj + dj, &c)) continue;
            float tt = (c.base + 40 - y) / sd[1], px = x + sd[0] * tt, pz = z + sd[2] * tt;
            for (int k = 0; k < c.n; k++) {
                float dx = px - c.x - c.p[k][0], dz = pz - c.z - c.p[k][1];
                float d = sqrtf(dx * dx + dz * dz) / c.p[k][2];
                cover = fmaxf(cover, 1 - smoothstep(0.65f, 1.0f, d));
            }
        }
    return 1 - 0.8f * cover;
}

static void build_tri(float *dst, const float p[3][3], const float raw[3], int32_t gx, int32_t gz, int tri, float lit) {
    float ux = p[1][0] - p[0][0], uy = p[1][1] - p[0][1], uz = p[1][2] - p[0][2];
    float vx = p[2][0] - p[0][0], vy = p[2][1] - p[0][1], vz = p[2][2] - p[0][2];
    float nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    if (ny < 0) { nx = -nx; ny = -ny; nz = -nz; }
    float nl = sqrtf(nx * nx + ny * ny + nz * nz);
    nx /= nl; ny /= nl; nz /= nl;

    float col[3];
    float jitter = 0.94f + 0.12f * rand01(gx * 2 + tri, gz, 99);
    int water = raw[0] < 0 && raw[1] < 0 && raw[2] < 0;
    if (water) {
        float deep = smoothstep(0, 120, -(raw[0] + raw[1] + raw[2]) / 3);
        col[0] = .20f - .08f * deep; col[1] = .42f - .14f * deep; col[2] = .66f - .10f * deep;
        jitter = 0.97f + 0.06f * rand01(gx * 2 + tri, gz, 98);
    } else {
        land_color((p[0][1] + p[1][1] + p[2][1]) / 3, 1 - ny, col);
    }
    float lam = nx * pal.sun_dir[0] + ny * pal.sun_dir[1] + nz * pal.sun_dir[2];
    if (lam < 0) lam = 0;
    lam *= lit;
    for (int k = 0; k < 3; k++) {
        float c = col[k] * (pal.ambient + lam * pal.light[k]) * jitter;
        col[k] = c > 1 ? 1 : c;
    }
    for (int i = 0; i < 3; i++) {
        float *o = dst + i * 6;
        o[0] = col[0]; o[1] = col[1]; o[2] = col[2];
        o[3] = p[i][0]; o[4] = p[i][1]; o[5] = p[i][2];
    }
}

static float ground(float x, float z);

static Cell *get_cell(int32_t gx, int32_t gz) {
    Cell *c = &cache[((gz & (RING - 1)) * RING) + (gx & (RING - 1))];
    if (c->gx == gx && c->gz == gz) return c;
    c->gx = gx; c->gz = gz;
    float x0 = gx * CELL, z0 = gz * CELL, x1 = x0 + CELL, z1 = z0 + CELL;
    float r00 = terrain(x0, z0), r10 = terrain(x1, z0), r01 = terrain(x0, z1), r11 = terrain(x1, z1);
    float h00 = fmaxf(r00, 0), h10 = fmaxf(r10, 0), h01 = fmaxf(r01, 0), h11 = fmaxf(r11, 0);
    float h = (h00 + h10 + h01 + h11) / 4, lit = c->lit = sun_lit(x0 + CELL / 2, h, z0 + CELL / 2);
    if ((gx + gz) & 1) {
        float a[3][3] = {{x0, h00, z0}, {x0, h01, z1}, {x1, h10, z0}}, ra[3] = {r00, r01, r10};
        float b[3][3] = {{x1, h10, z0}, {x0, h01, z1}, {x1, h11, z1}}, rb[3] = {r10, r01, r11};
        build_tri(&c->v[0][0], a, ra, gx, gz, 0, lit);
        build_tri(&c->v[3][0], b, rb, gx, gz, 1, lit);
    } else {
        float a[3][3] = {{x0, h00, z0}, {x1, h11, z1}, {x1, h10, z0}}, ra[3] = {r00, r11, r10};
        float b[3][3] = {{x0, h00, z0}, {x0, h01, z1}, {x1, h11, z1}}, rb[3] = {r00, r01, r11};
        build_tri(&c->v[0][0], a, ra, gx, gz, 0, lit);
        build_tri(&c->v[3][0], b, rb, gx, gz, 1, lit);
    }

    /* trees: none on beaches, steep slopes or above the treeline; clumped into
     * forests by a low-frequency mask; pines take over with height */
    c->nt = 0;
    float gxs = ((r10 - r00) + (r11 - r01)) / (2 * CELL), gzs = ((r01 - r00) + (r11 - r10)) / (2 * CELL);
    float slope = 1 - 1 / sqrtf(1 + gxs * gxs + gzs * gzs);
    if (opt_trees && fminf(fminf(r00, r10), fminf(r01, r11)) > 8 && slope < 0.35f) {
        float dens = h < 110 ? 0.3f : h < 230 ? 1.1f : h < 430 ? 2.2f : h < 600 ? 0.6f : 0;
        dens *= 1.7f * smoothstep(0.35f, 0.65f, fbm((x0 + seed_x) / 420, (z0 + seed_z) / 420, 3, 55));
        int n = (int)(dens + rand01(gx, gz, 60));
        if (n > MAXT) n = MAXT;
        for (int t = 0; t < n; t++) {
            float *p = c->tr[c->nt++];
            p[0] = x0 + CELL * (0.1f + 0.8f * rand01(gx * 4 + t, gz, 61));
            p[2] = z0 + CELL * (0.1f + 0.8f * rand01(gx * 4 + t, gz, 62));
            p[1] = ground(p[0], p[2]) - 1;
            float pine_odds = h > 230 ? 0.9f : h > 110 ? 0.4f : 0.1f;
            p[5] = rand01(gx * 4 + t, gz, 64) < pine_odds;
            p[3] = (p[5] ? 16 : 12) + 14 * rand01(gx * 4 + t, gz, 63);
            p[4] = (p[5] ? 0.3f : 0.42f) * p[3];
            p[6] = 0.85f + 0.3f * rand01(gx * 4 + t, gz, 65);
        }
    }
    return c;
}

/* Height of the drawn mesh (sea clamped to 0), for the autopilot. */
static float ground(float x, float z) {
    float fx = floorf(x / CELL), fz = floorf(z / CELL);
    float tx = x / CELL - fx, tz = z / CELL - fz;
    float x0 = fx * CELL, z0 = fz * CELL;
    float a = fmaxf(terrain(x0, z0), 0), b = fmaxf(terrain(x0 + CELL, z0), 0);
    float c = fmaxf(terrain(x0, z0 + CELL), 0), d = fmaxf(terrain(x0 + CELL, z0 + CELL), 0);
    return (a + (b - a) * tx) * (1 - tz) + (c + (d - c) * tx) * tz;
}

/* ---- flight model (an autopilot, really) ---- */
static struct {
    float x, y, z, yaw, pitch, roll, vy;
    float speed, turn, turn_target, clearance, next_change;
} fl;
static uint32_t rng_state;
static float frand(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (rng_state >> 8) / (float)0x1000000;
}

static void pick_route(float t) {
    fl.turn_target = (frand() * 2 - 1) * 0.22f;
    if (frand() < 0.3f) fl.turn_target *= 0.15f;           /* long straight legs */
    fl.clearance = 60 + frand() * frand() * 260;           /* mostly low */
    fl.next_change = t + 4 + frand() * 8;
}

static void fly(float dt, float t) {
    if (t > fl.next_change) pick_route(t);

    /* steer toward lower ground: compare terrain ahead-left vs ahead-right */
    float look = 450;
    float lx = fl.x - sinf(fl.yaw + 0.6f) * look, lz = fl.z - cosf(fl.yaw + 0.6f) * look;
    float rx = fl.x - sinf(fl.yaw - 0.6f) * look, rz = fl.z - cosf(fl.yaw - 0.6f) * look;
    float bias = (ground(rx, rz) - ground(lx, lz)) * 0.0006f;
    if (bias > 0.3f) bias = 0.3f; else if (bias < -0.3f) bias = -0.3f;
    float want = fl.turn_target + bias;
    if (want > 0.32f) want = 0.32f; else if (want < -0.32f) want = -0.32f;
    fl.turn += (want - fl.turn) * (1 - expf(-dt * 0.8f));
    fl.yaw += fl.turn * dt;

    float roll_target = fl.turn * 1.5f;
    fl.roll += (roll_target - fl.roll) * (1 - expf(-dt * 2.0f));

    /* terrain following: highest ground along the path ahead, discounted by distance */
    float need = -1e9f;
    for (float d = 0; d <= 1600; d += 50) {
        float h = ground(fl.x - sinf(fl.yaw + fl.turn * d / fl.speed) * d,
                         fl.z - cosf(fl.yaw + fl.turn * d / fl.speed) * d);
        float n = h + fl.clearance - d * 0.10f;
        if (n > need) need = n;
    }
    float floor_h = ground(fl.x, fl.z) + 18;
    if (need < floor_h) need = floor_h;
    float vy_target = (need - fl.y) * 0.5f;
    if (vy_target > 55) vy_target = 55; else if (vy_target < -30) vy_target = -30;
    fl.vy += (vy_target - fl.vy) * (1 - expf(-dt * 1.5f));
    fl.y += fl.vy * dt;
    if (fl.y < floor_h) fl.y = floor_h;
    float pitch_target = atan2f(fl.vy, fl.speed);
    fl.pitch += (pitch_target - fl.pitch) * (1 - expf(-dt * 2.5f));

    fl.x -= sinf(fl.yaw) * fl.speed * dt;
    fl.z -= cosf(fl.yaw) * fl.speed * dt;
}

/* ---- drawing ---- */
static int W = 800, H = 600;

/* The virtual camera: canvas point (cx, cy) is straight ahead, cam_f is the
 * focal length in canvas units, and this window shows canvas rect view_*.
 * Canvas y grows downward. Without --layout the canvas is just the window. */
static float cam_cx, cam_cy, cam_f, view_x, view_y, view_w, view_h, hud_h;

static void setup_view(void) {
    if (span_me < 0) {
        view_x = view_y = 0; view_w = W; view_h = H;
        cam_cx = W / 2.0f; cam_cy = H / 2.0f; hud_h = H;
        cam_f = H / 2.0f / tanf(opt_fov / 2 / DEG);
        return;
    }
    float x0 = 1e9f, x1 = -1e9f;
    for (int i = 0; i < span_n; i++) {
        x0 = fminf(x0, span[i][0]); x1 = fmaxf(x1, span[i][0] + span[i][2]);
    }
    /* centre on the output under the middle of the layout (the "front" one) */
    float mx = (x0 + x1) / 2;
    int mid = 0;
    for (int i = 0; i < span_n; i++)
        if (span[i][0] <= mx && mx < span[i][0] + span[i][2]) { mid = i; break; }
    cam_cx = span[mid][0] + span[mid][2] / 2;
    cam_cy = span[mid][1] + span[mid][3] / 2;
    hud_h = span[mid][3];
    /* --fov is the horizontal angle across the whole layout here */
    float halfw = fmaxf(cam_cx - x0, x1 - cam_cx);
    cam_f = halfw / tanf((opt_fov_set ? opt_fov : 120.0f) / 2 / DEG);
    view_x = span[span_me][0]; view_y = span[span_me][1];
    view_w = span[span_me][2]; view_h = span[span_me][3];
}

static void draw_sky(void) {
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_FOG);
    glDepthMask(GL_FALSE);
    glShadeModel(GL_SMOOTH);
    const float R = FAR * 0.8f;
    const int SEG = 36, RINGS = 10;
    for (int r = -2; r < RINGS; r++) {
        float e0 = r * (float)M_PI / 2 / RINGS, e1 = (r + 1) * (float)M_PI / 2 / RINGS;
        glBegin(GL_QUAD_STRIP);
        for (int s = 0; s <= SEG; s++) {
            float a = s * 2 * (float)M_PI / SEG;
            for (int k = 0; k < 2; k++) {
                float e = k ? e1 : e0;
                float t = e <= 0 ? 0 : powf(sinf(e), 0.55f);
                glColor3f(pal.sky_hor[0] + (pal.sky_zen[0] - pal.sky_hor[0]) * t,
                          pal.sky_hor[1] + (pal.sky_zen[1] - pal.sky_hor[1]) * t,
                          pal.sky_hor[2] + (pal.sky_zen[2] - pal.sky_hor[2]) * t);
                glVertex3f(fl.x + cosf(a) * cosf(e) * R, fl.y + sinf(e) * R, fl.z + sinf(a) * cosf(e) * R);
            }
        }
        glEnd();
    }
    /* sun: a flat disc on the dome */
    float sx = fl.x + pal.sun_dir[0] * R * 0.95f, sy = fl.y + pal.sun_dir[1] * R * 0.95f,
          sz = fl.z + pal.sun_dir[2] * R * 0.95f;
    float ux = -pal.sun_dir[2], uz = pal.sun_dir[0], ul = sqrtf(ux * ux + uz * uz);
    ux /= ul; uz /= ul;                       /* horizontal axis */
    float vx = -pal.sun_dir[1] * uz, vy = pal.sun_dir[2] * ux - pal.sun_dir[0] * uz, vz = pal.sun_dir[1] * ux;
    float rad = R * 0.035f;
    glBegin(GL_TRIANGLE_FAN);
    glColor3fv(pal.sun);
    glVertex3f(sx, sy, sz);
    for (int s = 0; s <= 24; s++) {
        float a = s * 2 * (float)M_PI / 24;
        glVertex3f(sx + (ux * cosf(a) + vx * sinf(a)) * rad, sy + vy * sinf(a) * rad,
                   sz + (uz * cosf(a) + vz * sinf(a)) * rad);
    }
    glEnd();
    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_FLAT);
}

static void draw_terrain(void) {
    int32_t cx = (int32_t)floorf(fl.x / CELL), cz = (int32_t)floorf(fl.z / CELL);
    float fx = -sinf(fl.yaw), fz = -cosf(fl.yaw);
    int n = 0;
    for (int32_t gz = cz - HALF; gz <= cz + HALF; gz++) {
        for (int32_t gx = cx - HALF; gx <= cx + HALF; gx++) {
            float dx = (gx - cx) * CELL, dz = (gz - cz) * CELL;
            if (dx * dx + dz * dz > FAR * FAR) continue;
            /* skip what's well behind us (keep a margin for the wide FOV + bank) */
            if (dx * fx + dz * fz < -3 * CELL && (dx * dx + dz * dz) > 36 * CELL * CELL) continue;
            memcpy(verts + n * 36, get_cell(gx, gz)->v, sizeof(float) * 36);
            n++;
        }
    }
    glEnable(GL_FOG);
    glInterleavedArrays(GL_C3F_V3F, 0, verts);
    glDrawArrays(GL_TRIANGLES, 0, n * 6);
}

/* One flat-shaded triangle, its normal flipped to face away from `in`. */
static float *emit(float *o, const float a[3], const float b[3], const float c[3],
                   const float col[3], const float in[3], float amb, float lit) {
    float ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
    float vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
    float nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    float nl = sqrtf(nx * nx + ny * ny + nz * nz);
    if (nl < 1e-6f) return o;
    float mx = (a[0] + b[0] + c[0]) / 3 - in[0], my = (a[1] + b[1] + c[1]) / 3 - in[1], mz = (a[2] + b[2] + c[2]) / 3 - in[2];
    if (nx * mx + ny * my + nz * mz < 0) nl = -nl;
    float lam = (nx * pal.sun_dir[0] + ny * pal.sun_dir[1] + nz * pal.sun_dir[2]) / nl;
    lam = lam < 0 ? 0 : lam * lit;
    const float *p[3] = {a, b, c};
    for (int i = 0; i < 3; i++, o += 6) {
        for (int k = 0; k < 3; k++) o[k] = fminf(1, col[k] * (amb + lam * pal.light[k]));
        o[3] = p[i][0]; o[4] = p[i][1]; o[5] = p[i][2];
    }
    return o;
}

/* 5-sided cone from a ring at yb to the apex at ya (ya < yb points it down). */
static float *cone(float *o, float x, float z, float r, float yb, float ya, float rot, const float col[3], float lit) {
    float apex[3] = {x, ya, z}, in[3] = {x, yb + (ya - yb) * 0.25f, z}, ring[6][3];
    for (int s = 0; s <= 5; s++) {
        float a = rot + s * 2 * (float)M_PI / 5;
        ring[s][0] = x + cosf(a) * r; ring[s][1] = yb; ring[s][2] = z + sinf(a) * r;
    }
    for (int s = 0; s < 5; s++) o = emit(o, ring[s], ring[s + 1], apex, col, in, pal.ambient, lit);
    return o;
}

static float *trunk(float *o, float x, float z, float r, float y0, float y1, float rot, float lit) {
    static const float col[3] = {.36f, .26f, .16f};
    float in[3] = {x, (y0 + y1) / 2, z}, b[4][3], t[4][3];
    for (int s = 0; s <= 3; s++) {
        float a = rot + s * 2 * (float)M_PI / 3;
        b[s][0] = t[s][0] = x + cosf(a) * r; b[s][2] = t[s][2] = z + sinf(a) * r;
        b[s][1] = y0; t[s][1] = y1;
    }
    for (int s = 0; s < 3; s++) {
        o = emit(o, b[s], b[s + 1], t[s], col, in, pal.ambient, lit);
        o = emit(o, b[s + 1], t[s + 1], t[s], col, in, pal.ambient, lit);
    }
    return o;
}

#define TREE_R 96                /* cells; trees grow in over the last 20% */
#define MAX_TREES 30000
#define TREE_TRIS 16
static float *tverts;

static void draw_trees(void) {
    static const float pine[3] = {.11f, .28f, .14f}, leaf[3] = {.24f, .44f, .15f};
    int32_t cx = (int32_t)floorf(fl.x / CELL), cz = (int32_t)floorf(fl.z / CELL);
    float fx = -sinf(fl.yaw), fz = -cosf(fl.yaw), R = TREE_R * CELL;
    float *o = tverts;
    int count = 0;
    for (int32_t gz = cz - TREE_R; gz <= cz + TREE_R; gz++) {
        for (int32_t gx = cx - TREE_R; gx <= cx + TREE_R; gx++) {
            float dx = (gx - cx) * CELL, dz = (gz - cz) * CELL, d2 = dx * dx + dz * dz;
            if (d2 > R * R) continue;
            if (dx * fx + dz * fz < -3 * CELL && d2 > 36 * CELL * CELL) continue;
            Cell *c = get_cell(gx, gz);
            if (!c->nt) continue;
            float fade = smoothstep(R, R * 0.8f, sqrtf(d2));
            for (int t = 0; t < c->nt && count < MAX_TREES; t++, count++) {
                const float *p = c->tr[t];
                float x = p[0], y = p[1], z = p[2], h = p[3] * fade, r = p[4] * fade, rot = x * 0.37f + z * 0.61f;
                if (h < 0.5f) continue;
                const float *base = p[5] ? pine : leaf;
                float col[3] = {base[0] * p[6], base[1] * p[6], base[2] * p[6]};
                if (p[5]) {
                    o = trunk(o, x, z, r * 0.14f, y, y + 0.25f * h, rot, c->lit);
                    o = cone(o, x, z, r, y + 0.15f * h, y + 0.72f * h, rot, col, c->lit);
                    o = cone(o, x, z, r * 0.68f, y + 0.48f * h, y + h, rot + 0.6f, col, c->lit);
                } else {
                    o = trunk(o, x, z, r * 0.12f, y, y + 0.4f * h, rot, c->lit);
                    o = cone(o, x, z, r, y + 0.62f * h, y + h, rot, col, c->lit);
                    o = cone(o, x, z, r, y + 0.62f * h, y + 0.28f * h, rot, col, c->lit);
                }
            }
        }
    }
    glInterleavedArrays(GL_C3F_V3F, 0, tverts);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)((o - tverts) / 6));
}

#define CLOUD_R ((int)(FAR / CG) + 2)
#define PSEG 8
#define PUFF_TRIS (PSEG * 6)
static float *cverts;

/* A puff: a jittered, flattened dome (staggered rings) on a flat base. */
static float *puff(float *o, const Cloud *c, int k, int32_t i, int32_t j, float amb) {
    static const float col[3] = {.96f, .97f, 1.0f};
    float cx = c->x + c->p[k][0], cz = c->z + c->p[k][1], r = c->p[k][2], hs = c->p[k][3], y0 = c->base;
    float v[3][PSEG + 1][3];
    for (int ring = 0; ring < 3; ring++) {
        float e = ring * (float)M_PI / 6;
        for (int s = 0; s < PSEG; s++) {
            float jit = 0.82f + 0.3f * rand01(i * 64 + k * 8 + ring, j * 16 + s, 220);
            float a = (s + 0.5f * (ring & 1)) * 2 * (float)M_PI / PSEG, rr = r * cosf(e) * jit;
            v[ring][s][0] = cx + cosf(a) * rr;
            v[ring][s][1] = y0 + r * hs * sinf(e) * jit;
            v[ring][s][2] = cz + sinf(a) * rr;
        }
        memcpy(v[ring][PSEG], v[ring][0], sizeof v[ring][0]);
    }
    float apex[3] = {cx, y0 + r * hs * (0.95f + 0.1f * rand01(i * 64 + k, j, 221)), cz};
    float mid[3] = {cx, y0, cz}, in[3] = {cx, y0 + r * hs * 0.3f, cz};
    for (int s = 0; s < PSEG; s++) {
        o = emit(o, v[0][s], v[0][s + 1], mid, col, in, amb, 1);
        for (int ring = 0; ring < 2; ring++) {
            o = emit(o, v[ring][s], v[ring][s + 1], v[ring + 1][s], col, in, amb, 1);
            o = emit(o, v[ring][s + 1], v[ring + 1][s + 1], v[ring + 1][s], col, in, amb, 1);
        }
        o = emit(o, v[2][s], v[2][s + 1], apex, col, in, amb, 1);
    }
    return o;
}

static void draw_clouds(void) {
    int32_t ci = (int32_t)floorf(fl.x / CG), cj = (int32_t)floorf(fl.z / CG);
    float fx = -sinf(fl.yaw), fz = -cosf(fl.yaw), amb = pal.ambient + 0.3f;
    float *o = cverts;
    for (int32_t j = cj - CLOUD_R; j <= cj + CLOUD_R; j++) {
        for (int32_t i = ci - CLOUD_R; i <= ci + CLOUD_R; i++) {
            Cloud c;
            if (!get_cloud(i, j, &c)) continue;
            float dx = c.x - fl.x, dz = c.z - fl.z, d2 = dx * dx + dz * dz;
            if (d2 > (FAR + 400) * (FAR + 400)) continue;
            if (dx * fx + dz * fz < -600 && d2 > 800 * 800) continue;
            for (int k = 0; k < c.n; k++) o = puff(o, &c, k, i, j, amb);
        }
    }
    glInterleavedArrays(GL_C3F_V3F, 0, cverts);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)((o - cverts) / 6));
}

/* 7-segment vector digits for the HUD */
static void seg_digit(float x, float y, float w, float h, int d) {
    static const unsigned char segs[10] = {0x3f, 0x06, 0x5b, 0x4f, 0x66, 0x6d, 0x7d, 0x07, 0x7f, 0x6f};
    const float pts[7][4] = {{0, 1, 1, 1}, {1, 1, 1, .5f}, {1, .5f, 1, 0}, {0, 0, 1, 0},
                             {0, 0, 0, .5f}, {0, .5f, 0, 1}, {0, .5f, 1, .5f}};
    for (int s = 0; s < 7; s++)
        if (segs[d] & (1 << s)) {
            glVertex2f(x + pts[s][0] * w, y + pts[s][1] * h);
            glVertex2f(x + pts[s][2] * w, y + pts[s][3] * h);
        }
}
static void seg_number(float x, float y, float h, int v, int min_digits, int right_align) {
    char buf[16];
    snprintf(buf, sizeof buf, "%0*d", min_digits, v < 0 ? -v : v);
    int len = (int)strlen(buf);
    float w = h * 0.5f, adv = h * 0.8f;
    if (right_align) x -= len * adv - (adv - w);
    glBegin(GL_LINES);
    for (int i = 0; i < len; i++) seg_digit(x + i * adv, y, w, h, buf[i] - '0');
    glEnd();
}

static void draw_hud(void) {
    float s = hud_h / 720.0f;
    /* canvas units, origin at the camera centre, y up */
    float ox = view_x - cam_cx, oy = cam_cy - view_y - view_h, px = W / view_w, py = H / view_h;
    glMatrixMode(GL_PROJECTION); glLoadIdentity();
    glOrtho(ox, ox + view_w, oy, oy + view_h, -1, 1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    glDisable(GL_DEPTH_TEST); glDisable(GL_FOG);
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);
    glLineWidth(fmaxf(1.5f, 2.2f * s * px));
    glColor4f(0.35f, 1.0f, 0.45f, 0.85f);

    /* pitch ladder, clipped to the middle of the screen, rolled with the horizon */
    float px_per_deg = cam_f / DEG;
    glEnable(GL_SCISSOR_TEST);
    glScissor((int)((-260 * s - ox) * px), (int)((-230 * s - oy) * py), (int)(520 * s * px), (int)(460 * s * py));
    glPushMatrix();
    glRotatef(-fl.roll * DEG, 0, 0, 1);
    glTranslatef(0, -fl.pitch * DEG * px_per_deg, 0);
    for (int p = -40; p <= 40; p += 10) {
        float y = p * px_per_deg;
        float half = (p == 0 ? 220 : 110) * s, gap = 45 * s;
        glBegin(GL_LINES);
        if (p < 0) { /* dashed below the horizon */
            for (float x = gap; x < half; x += 22 * s) {
                glVertex2f(x, y); glVertex2f(fminf(x + 12 * s, half), y);
                glVertex2f(-x, y); glVertex2f(-fminf(x + 12 * s, half), y);
            }
        } else {
            glVertex2f(gap, y); glVertex2f(half, y);
            glVertex2f(-gap, y); glVertex2f(-half, y);
        }
        if (p != 0) { /* end ticks point toward the horizon */
            float tk = (p > 0 ? -10 : 10) * s;
            glVertex2f(half, y); glVertex2f(half, y + tk);
            glVertex2f(-half, y); glVertex2f(-half, y + tk);
        }
        glEnd();
        if (p != 0) {
            seg_number(half + 8 * s, y - 7 * s, 14 * s, p < 0 ? -p : p, 1, 0);
            seg_number(-half - 8 * s, y - 7 * s, 14 * s, p < 0 ? -p : p, 1, 1);
        }
    }
    glPopMatrix();
    glDisable(GL_SCISSOR_TEST);

    /* boresight "W" */
    glBegin(GL_LINE_STRIP);
    glVertex2f(-34 * s, 0); glVertex2f(-16 * s, 0); glVertex2f(-8 * s, -10 * s); glVertex2f(0, 0);
    glVertex2f(8 * s, -10 * s); glVertex2f(16 * s, 0); glVertex2f(34 * s, 0);
    glEnd();

    /* heading tape */
    float hdg = fmodf(-fl.yaw * DEG, 360); if (hdg < 0) hdg += 360;
    float top = hud_h / 2 - 60 * s, span = 150 * s, ppd = 7 * s;
    glBegin(GL_LINES);
    for (int d = (int)floorf(hdg - span / ppd); d <= (int)ceilf(hdg + span / ppd); d++) {
        if (d % 5) continue;
        float x = (d - hdg) * ppd;
        if (fabsf(x) > span) continue;
        glVertex2f(x, top); glVertex2f(x, top + (d % 10 ? 7 : 13) * s);
    }
    glVertex2f(0, top - 4 * s); glVertex2f(-6 * s, top - 14 * s);
    glVertex2f(0, top - 4 * s); glVertex2f(6 * s, top - 14 * s);
    glEnd();
    for (int d = (int)floorf((hdg - span / ppd) / 10) * 10; d <= hdg + span / ppd; d += 10) {
        float x = (d - hdg) * ppd;
        if (fabsf(x) > span - 10 * s) continue;
        int lab = ((d % 360) + 360) % 360 / 10;
        seg_number(x - 9 * s, top + 18 * s, 13 * s, lab, 2, 0);
    }

    /* airspeed (left) and altitude (right) boxes */
    float bx = 300 * s, bw = 92 * s, bh = 34 * s;
    for (int side = -1; side <= 1; side += 2) {
        float x0 = side < 0 ? -bx - bw : bx, x1 = x0 + bw;
        glBegin(GL_LINE_LOOP);
        glVertex2f(x0, -bh / 2); glVertex2f(x1, -bh / 2); glVertex2f(x1, bh / 2); glVertex2f(x0, bh / 2);
        glEnd();
        int v = side < 0 ? (int)(fl.speed * 1.9f) : (int)(fl.y * 3.28f);
        seg_number(x1 - 10 * s, -9 * s, 18 * s, v, side < 0 ? 3 : 4, 1);
    }
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
}

static void render(void) {
    setup_view();
    glViewport(0, 0, W, H);
    glClearColor(pal.sky_hor[0], pal.sky_hor[1], pal.sky_hor[2], 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION); glLoadIdentity();
    float nearp = 2, k = nearp / cam_f;
    glFrustum((view_x - cam_cx) * k, (view_x + view_w - cam_cx) * k,
              (cam_cy - view_y - view_h) * k, (cam_cy - view_y) * k, nearp, FAR * 1.1f);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    glRotatef(-fl.roll * DEG, 0, 0, 1);
    glRotatef(-fl.pitch * DEG, 1, 0, 0);
    glRotatef(-fl.yaw * DEG, 0, 1, 0);
    glTranslatef(-fl.x, -fl.y, -fl.z);

    draw_sky();
    draw_terrain();
    if (opt_trees) draw_trees();
    if (opt_clouds) draw_clouds();
    if (opt_hud) draw_hud();
}

static void choose_palette(void) {
    int dusk = !strcmp(opt_time, "dusk") || (!strcmp(opt_time, "random") && (opt_seed & 1));
    float el, az;
    if (dusk) {
        pal = (Palette){{.16f, .14f, .38f}, {.98f, .62f, .38f}, {1, .86f, .55f}, {0}, {1.05f, .72f, .52f}, .30f};
        el = 7; az = 200;
    } else {
        pal = (Palette){{.18f, .38f, .80f}, {.72f, .84f, .95f}, {1, 1, .92f}, {0}, {.85f, .85f, .80f}, .42f};
        el = 38; az = 150;
    }
    pal.sun_dir[0] = cosf(el / DEG) * sinf(az / DEG);
    pal.sun_dir[1] = sinf(el / DEG);
    pal.sun_dir[2] = cosf(el / DEG) * cosf(az / DEG);
    float fog[4] = {pal.sky_hor[0], pal.sky_hor[1], pal.sky_hor[2], 1};
    glFogi(GL_FOG_MODE, GL_LINEAR);
    glFogfv(GL_FOG_COLOR, fog);
    glFogf(GL_FOG_START, FAR * 0.30f);
    glFogf(GL_FOG_END, FAR * 0.97f);
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static double wallclock(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* "x,y,w,h;x,y,w,h;..." */
static void parse_layout(const char *str) {
    for (span_n = 0; span_n < MAX_OUT; span_n++) {
        float *r = span[span_n];
        int used = 0;
        if (sscanf(str, "%f,%f,%f,%f%n", &r[0], &r[1], &r[2], &r[3], &used) != 4 || r[2] <= 0 || r[3] <= 0)
            break;
        str += used;
        if (*str == ';') str++;
    }
}

static void usage(void) {
    fprintf(stderr,
        "usage: flightsim [--window | --root] [--no-hud] [--no-trees] [--no-clouds]\n"
        "                 [--time day|dusk|random]\n"
        "                 [--speed X] [--fov DEG] [--seed N] [--epoch UNIXTIME]\n"
        "                 [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n");
    exit(1);
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (a[0] == '-' && a[1] == '-') a++;                /* accept -x and --x */
        if (!strcmp(a, "-root")) opt_root = 1;
        else if (!strcmp(a, "-window")) opt_root = 0;
        else if (!strcmp(a, "-no-hud")) opt_hud = 0;
        else if (!strcmp(a, "-hud")) opt_hud = 1;
        else if (!strcmp(a, "-no-trees")) opt_trees = 0;
        else if (!strcmp(a, "-no-clouds")) opt_clouds = 0;
        else if (!strcmp(a, "-time") && i + 1 < argc) opt_time = argv[++i];
        else if (!strcmp(a, "-speed") && i + 1 < argc) opt_speed = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-fov") && i + 1 < argc) { opt_fov = strtof(argv[++i], NULL); opt_fov_set = 1; }
        else if (!strcmp(a, "-epoch") && i + 1 < argc) opt_epoch = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-layout") && i + 1 < argc) parse_layout(argv[++i]);
        else if (!strcmp(a, "-output") && i + 1 < argc) span_me = atoi(argv[++i]);
        else if (!strcmp(a, "-seed") && i + 1 < argc) opt_seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else usage();
    }
    if (span_me >= span_n) span_me = -1;
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    rng_state = opt_seed;
    seed_x = (frand() - 0.5f) * 2e5f;
    seed_z = (frand() - 0.5f) * 2e5f;
    cloud_off_x = (int32_t)(seed_x / CG); cloud_off_z = (int32_t)(seed_z / CG);

    Display *dpy = XOpenDisplay(NULL);
    if (!dpy) { fprintf(stderr, "flightsim: can't open display\n"); return 1; }
    int scr = DefaultScreen(dpy);
    Window root = RootWindow(dpy, scr);
    int attrs[] = {GLX_RGBA, GLX_DOUBLEBUFFER, GLX_DEPTH_SIZE, 24, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, None};
    XVisualInfo *vi = glXChooseVisual(dpy, scr, attrs);
    if (!vi) { attrs[3] = 16; vi = glXChooseVisual(dpy, scr, attrs); }
    if (!vi) { fprintf(stderr, "flightsim: no suitable GLX visual\n"); return 1; }

    XWindowAttributes ra;
    XGetWindowAttributes(dpy, root, &ra);
    XSetWindowAttributes swa = {0};
    swa.colormap = XCreateColormap(dpy, root, vi->visual, AllocNone);
    swa.event_mask = StructureNotifyMask | KeyPressMask | ExposureMask;
    swa.override_redirect = opt_root;
    if (opt_root) { W = ra.width; H = ra.height; XSelectInput(dpy, root, StructureNotifyMask); }
    Window win = XCreateWindow(dpy, root, 0, 0, W, H, 0, vi->depth, InputOutput, vi->visual,
                               CWColormap | CWEventMask | CWOverrideRedirect, &swa);
    XStoreName(dpy, win, "Flight Sim");
    XClassHint ch = {"flightsim", "Flightsim"};
    XSetClassHint(dpy, win, &ch);
    Atom wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(dpy, win, &wm_delete, 1);
    XMapRaised(dpy, win);

    GLXContext ctx = glXCreateContext(dpy, vi, NULL, True);
    glXMakeCurrent(dpy, win, ctx);

    cache = malloc(sizeof(Cell) * RING * RING);
    for (int i = 0; i < RING * RING; i++) cache[i].gx = cache[i].gz = INT32_MIN;
    verts = malloc(sizeof(float) * 36 * (2 * HALF + 1) * (2 * HALF + 1));
    tverts = malloc(sizeof(float) * 18 * TREE_TRIS * MAX_TREES);
    cverts = malloc(sizeof(float) * 18 * PUFF_TRIS * PUFFS * (2 * CLOUD_R + 1) * (2 * CLOUD_R + 1));

    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_FLAT);
    choose_palette();

    fl.speed = 150 * opt_speed;
    fl.yaw = frand() * 2 * (float)M_PI;
    fl.y = ground(0, 0) + 150;
    pick_route(0);

    /* Fixed-step sim clocked off the wall: the state at time t depends only on
     * seed + epoch + t, so separate instances stay in lockstep. */
    const double STEP = 1.0 / 120;
    double epoch = opt_epoch ? opt_epoch : wallclock();
    long steps = 0;
    for (;;) {
        while (XPending(dpy)) {
            XEvent ev;
            XNextEvent(dpy, &ev);
            if (ev.type == ConfigureNotify) {
                if (ev.xconfigure.window == root) XResizeWindow(dpy, win, ev.xconfigure.width, ev.xconfigure.height);
                else { W = ev.xconfigure.width; H = ev.xconfigure.height; }
            } else if (ev.type == KeyPress) {
                KeySym k = XLookupKeysym(&ev.xkey, 0);
                if (k == XK_q || k == XK_Escape) return 0;
            } else if (ev.type == ClientMessage && (Atom)ev.xclient.data.l[0] == wm_delete) {
                return 0;
            }
        }
        double t = now();
        double sim_t = wallclock() - epoch;
        while ((steps + 1) * STEP <= sim_t) {
            fly((float)STEP, (float)(steps * STEP));
            steps++;
        }
        render();
        glXSwapBuffers(dpy, win);
        double spare = 1.0 / 60 - (now() - t);
        if (spare > 0) {
            struct timespec ts = {0, (long)(spare * 1e9)};
            nanosleep(&ts, NULL);
        }
    }
}
