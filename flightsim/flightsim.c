/*
 * flightsim — a flight-sim screensaver.
 *
 * Endless procedural mountains streamed into GPU-tessellated terrain (~14 km of
 * view), lit per pixel with baked terrain shadows and ambient occlusion,
 * height fog that pools in the valleys, raymarched cumulus that shade the
 * ground, reflective water, instanced forests, a painted sky with stars at
 * night, HDR bloom and filmic tonemapping. An autopilot banks through valleys
 * and hops ridges under the cloud deck, with an optional green vector HUD.
 *
 * Plain Xlib + GLX (GL 4.5 compatibility context), like the xscreensaver
 * hacks, so it runs the same way: --window, or --root (a full-size child of
 * the root window, which is what the swaylock-plugin / Xwayland chain needs).
 * The scene renders at a dynamic resolution that tracks the frame budget, so
 * several instances can share a small iGPU.
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
#define GL_GLEXT_PROTOTYPES
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glx.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define DEG (180.0f / (float)M_PI)

/* ---- options ---- */
static int opt_root = 0, opt_hud = 1;
static float opt_speed = 1.0f, opt_fov = 60.0f, opt_scale = 0;   /* 0 = dynamic */
static const char *opt_time = "random";
static uint32_t opt_seed = 0;
static double opt_epoch = 0;          /* CLOCK_REALTIME seconds the sim starts at */
static int opt_fov_set = 0;
static int opt_trees = 1, opt_clouds = 1;
static const char *opt_shot;          /* render one frame offscreen to this PPM */
static int opt_bench;                 /* render this many frames offscreen, print timing */
static int off_w = 1920, off_h = 1080;
static double opt_at = 0;             /* start the sim this many seconds in */

/* ---- spanning ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];       /* x, y, w, h of each output (logical) */
static int span_n = 0, span_me = -1;

/* ---- noise ---- */
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

/* ---- terrain ---- */
/* Raw height; negative = under the sea. Domain-warped hills, with ridged
 * mountain ranges up to ~1.7 km where a low-frequency mask allows. */
static float terrain(double x, double z) {
    x += seed_x; z += seed_z;
    float wx = fbm((float)(x / 2400), (float)(z / 2400), 3, 71) - 0.5f;
    float wz = fbm((float)(x / 2400) + 5.2f, (float)(z / 2400) + 1.3f, 3, 72) - 0.5f;
    double X = x + wx * 1400, Z = z + wz * 1400;
    float base = fbm((float)(X / 1100), (float)(Z / 1100), 5, 11) * 300;
    float mask = smoothstep(0.40f, 0.66f, fbm((float)(x / 4200), (float)(z / 4200), 3, 23));
    float mtn = 0;
    if (mask > 0) mtn = powf(ridged((float)(X / 950), (float)(Z / 950), 6, 37), 1.35f) * 1900 * mask;
    return base + mtn - 110;
}

/* How wooded the ground is (0..1), before slope is considered. */
static float forest(double x, double z, float h) {
    if (h < 5 || h > 1250) return 0;
    float alt = smoothstep(5, 30, h) * (1 - smoothstep(950, 1250, h));
    float m = smoothstep(0.42f, 0.60f, fbm((float)((x + seed_x) / 650), (float)((z + seed_z) / 650), 4, 55));
    float low = 0.55f + 0.45f * smoothstep(60, 220, h);    /* meadows down low */
    return alt * m * low;
}

#define BASE 8.0                     /* metres per finest lattice cell */

/* Height of the ground (sea clamped to 0), bilinear over the finest lattice. */
static float ground(double x, double z) {
    double fx = floor(x / BASE), fz = floor(z / BASE);
    float tx = (float)(x / BASE - fx), tz = (float)(z / BASE - fz);
    double x0 = fx * BASE, z0 = fz * BASE;
    float a = fmaxf(terrain(x0, z0), 0), b = fmaxf(terrain(x0 + BASE, z0), 0);
    float c = fmaxf(terrain(x0, z0 + BASE), 0), d = fmaxf(terrain(x0 + BASE, z0 + BASE), 0);
    return (a + (b - a) * tx) * (1 - tz) + (c + (d - c) * tx) * tz;
}

/* ---- time of day ---- */
typedef struct {
    const char *name;
    float el, az, sun[3], sun_i, zen[3], hor[3], hor_sun[3];
    float fog_a, fog_b, haze;        /* height fog: density at sea level, falloff (1/m); uniform haze */
    float cover, cbase, cthick, cdens;
    float exposure, glow, stars, disc, amb, sat;
} Preset;

/* colours are sRGB; converted to linear when chosen */
static const Preset presets[] = {
    {"day",    50, 140, {1, .96f, .90f}, 4.2f, {.20f, .42f, .82f}, {.70f, .81f, .93f}, {.86f, .88f, .92f},
     9e-5f, 1 / 1100.f, 1.1e-5f, .42f, 1500, 900, .06f, 1.0f, 1.0f, 0, .011f, 1.0f, 1.0f},
    {"golden",  9,  95, {1, .76f, .50f}, 3.6f, {.30f, .44f, .72f}, {.90f, .80f, .72f}, {1, .78f, .52f},
     2.4e-4f, 1 / 330.f, 1.4e-5f, .34f, 1550, 750, .05f, 1.05f, 1.4f, 0, .013f, 1.0f, 1.05f},
    {"dusk",  3.2f, 250, {1, .50f, .24f}, 3.0f, {.11f, .14f, .34f}, {.82f, .48f, .40f}, {1, .58f, .30f},
     1.3e-4f, 1 / 650.f, 1.3e-5f, .46f, 1550, 850, .05f, 1.2f, 1.7f, .12f, .015f, 1.0f, 1.05f},
    {"night",  36, 205, {.62f, .72f, 1}, .30f, {.014f, .024f, .06f}, {.07f, .09f, .15f}, {.09f, .11f, .17f},
     8e-5f, 1 / 1000.f, 9e-6f, .30f, 1550, 750, .05f, 2.6f, .5f, 1, .010f, 1.6f, 0.45f},
};
#define NPRESET (int)(sizeof presets / sizeof presets[0])

static struct {
    float sun_dir[3], sun[3], zen[3], hor[3], hor_sun[3], amb_sky[3], amb_gnd[3];
    float fog_a, fog_b, haze, cover, cbase, cthick, cdens, exposure, glow, stars, disc, sat;
    float mist;                      /* valley mist density, 1/m */
    float shafts;                    /* light shaft strength */
} pal;

static float lin(float c) { return powf(c, 2.2f); }

static void choose_palette(void) {
    int k = 0;
    for (int i = 0; i < NPRESET; i++)
        if (!strcmp(opt_time, presets[i].name)) k = i + 1;
    k = k ? k - 1 : (int)(hash2((int32_t)opt_seed, 7, 99) % NPRESET);
    const Preset *p = &presets[k];
    float el = p->el / DEG, az = p->az / DEG;
    pal.sun_dir[0] = cosf(el) * sinf(az);
    pal.sun_dir[1] = sinf(el);
    pal.sun_dir[2] = cosf(el) * cosf(az);
    for (int i = 0; i < 3; i++) {
        pal.sun[i] = lin(p->sun[i]) * p->sun_i;
        pal.zen[i] = lin(p->zen[i]);
        pal.hor[i] = lin(p->hor[i]);
        pal.hor_sun[i] = lin(p->hor_sun[i]);
        pal.amb_sky[i] = (pal.zen[i] * 0.6f + pal.hor[i] * 0.4f) * 1.1f * p->amb;
        pal.amb_gnd[i] = (pal.hor[i] * 0.25f + pal.sun[i] * 0.03f * fmaxf(pal.sun_dir[1], 0)) * p->amb;
    }
    pal.fog_a = p->fog_a; pal.fog_b = p->fog_b; pal.haze = p->haze;
    pal.cover = p->cover; pal.cbase = p->cbase; pal.cthick = p->cthick; pal.cdens = p->cdens;
    pal.exposure = p->exposure; pal.glow = p->glow; pal.stars = p->stars; pal.disc = p->disc; pal.sat = p->sat;
    /* valley mist: thick with the sun low, lingering at night, wisps by day */
    pal.mist = p->stars > 0.5f ? 0.0009f : 0.0003f + 0.0008f * (1 - smoothstep(4, 25, p->el));
    /* light shafts: strongest with the sun low, faint by moonlight */
    pal.shafts = p->stars > 0.5f ? 0.15f : 0.35f + 0.65f * (1 - smoothstep(5, 30, p->el));
}

/* ---- flight model (an autopilot, really) ---- */
static struct {
    double x, z;
    float y, yaw, pitch, roll, vy;
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
    fl.clearance = 70 + frand() * frand() * 300;           /* mostly low */
    fl.next_change = t + 4 + frand() * 8;
}

static void fly(float dt, float t) {
    if (t > fl.next_change) pick_route(t);

    /* steer toward lower ground: compare terrain ahead-left vs ahead-right */
    float look = 500;
    double lx = fl.x - sinf(fl.yaw + 0.6f) * look, lz = fl.z - cosf(fl.yaw + 0.6f) * look;
    double rx = fl.x - sinf(fl.yaw - 0.6f) * look, rz = fl.z - cosf(fl.yaw - 0.6f) * look;
    float bias = (ground(rx, rz) - ground(lx, lz)) * 0.0006f;
    if (bias > 0.3f) bias = 0.3f; else if (bias < -0.3f) bias = -0.3f;
    float want = fl.turn_target + bias;
    if (want > 0.32f) want = 0.32f; else if (want < -0.32f) want = -0.32f;
    fl.turn += (want - fl.turn) * (1 - expf(-dt * 0.8f));
    fl.yaw += fl.turn * dt;

    float roll_target = fl.turn * 1.5f;
    fl.roll += (roll_target - fl.roll) * (1 - expf(-dt * 2.0f));

    /* terrain following: highest ground along the path ahead, discounted by
     * distance; stay under the cloud deck unless the ground forces us up */
    float need = -1e9f, safe = -1e9f;
    for (float d = 0; d <= 1600; d += 50) {
        float h = ground(fl.x - sinf(fl.yaw + fl.turn * d / fl.speed) * d,
                         fl.z - cosf(fl.yaw + fl.turn * d / fl.speed) * d);
        need = fmaxf(need, h + fl.clearance - d * 0.10f);
        safe = fmaxf(safe, h + 50 - d * 0.10f);
    }
    need = fmaxf(fminf(need, pal.cbase - 120), safe);
    float floor_h = ground(fl.x, fl.z) + 35;
    if (need < floor_h) need = floor_h;
    float vy_target = (need - fl.y) * 0.5f;
    if (vy_target > 55) vy_target = 55; else if (vy_target < -30) vy_target = -30;
    fl.vy += (vy_target - fl.vy) * (1 - expf(-dt * 1.5f));
    fl.y += fl.vy * dt;
    if (fl.y < floor_h) fl.y = floor_h;
    float pitch_target = atan2f(fl.vy, fl.speed);
    fl.pitch += (pitch_target - fl.pitch) * (1 - expf(-dt * 2.5f));

    fl.x -= sin(fl.yaw) * fl.speed * dt;
    fl.z -= cos(fl.yaw) * fl.speed * dt;
}

/* ---- threads ---- */
static int nthreads = 1;
typedef struct { void (*fn)(int, void *); void *arg; int n, next; pthread_mutex_t mu; } Job;
static void *job_worker(void *p) {
    Job *j = p;
    for (;;) {
        pthread_mutex_lock(&j->mu);
        int i = j->next++;
        pthread_mutex_unlock(&j->mu);
        if (i >= j->n) return NULL;
        j->fn(i, j->arg);
    }
}
/* fn(i) for i in [0, n), across all cores */
static void parallel_for(int n, void (*fn)(int, void *), void *arg) {
    Job j = {fn, arg, n, 0, PTHREAD_MUTEX_INITIALIZER};
    pthread_t th[64];
    int k = nthreads < 64 ? nthreads : 64;
    for (int i = 1; i < k; i++) pthread_create(&th[i], NULL, job_worker, &j);
    job_worker(&j);
    for (int i = 1; i < k; i++) pthread_join(th[i], NULL);
}

/* ---- streamed heightmaps ---- */
/* Toroidal textures: texel (i & (n-1), j & (n-1)) holds lattice point (i, j),
 * i.e. world (i, j) * unit * BASE, and GL_REPEAT does the wrapping for free.
 * R = height, G = forest. Only rows/columns that scroll in get computed. */
typedef struct {
    int n, unit;
    int32_t ox, oz;               /* lattice coords of the window's low corner */
    int valid;
    float *data;
    GLuint tex;
} HMap;
static HMap hm[2] = {{1024, 1, 0, 0, 0, NULL, 0}, {1024, 4, 0, 0, 0, NULL, 0}};

static void hm_texel(HMap *m, int32_t i, int32_t j) {
    double x = (double)i * m->unit * BASE, z = (double)j * m->unit * BASE;
    float h = terrain(x, z);
    float *d = m->data + 2 * ((size_t)(j & (m->n - 1)) * m->n + (i & (m->n - 1)));
    d[0] = h; d[1] = forest(x, z, h);
}
static void hm_fill_row(int r, void *arg) {
    HMap *m = arg;
    for (int32_t i = m->ox; i < m->ox + m->n; i++) hm_texel(m, i, m->oz + r);
}

static void hm_update(HMap *m, double cx, double cz) {
    double sp = m->unit * BASE;
    int n = m->n, mask = n - 1;
    int32_t nx = (int32_t)floor(cx / sp) - n / 2, nz = (int32_t)floor(cz / sp) - n / 2;
    glBindTexture(GL_TEXTURE_2D, m->tex);
    if (!m->valid || abs(nx - m->ox) >= n / 4 || abs(nz - m->oz) >= n / 4) {
        m->ox = nx; m->oz = nz; m->valid = 1;
        parallel_for(n, hm_fill_row, m);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, n, n, GL_RG, GL_FLOAT, m->data);
        glGenerateMipmap(GL_TEXTURE_2D);
        return;
    }
    glPixelStorei(GL_UNPACK_ROW_LENGTH, n);
    for (int32_t i = nx; i < nx + n; i++) {
        if (i >= m->ox && i < m->ox + n) continue;
        for (int32_t j = nz; j < nz + n; j++) hm_texel(m, i, j);
        glPixelStorei(GL_UNPACK_SKIP_PIXELS, i & mask);
        glTexSubImage2D(GL_TEXTURE_2D, 0, i & mask, 0, 1, n, GL_RG, GL_FLOAT, m->data);
    }
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
    for (int32_t j = nz; j < nz + n; j++) {
        if (j >= m->oz && j < m->oz + n) continue;
        for (int32_t i = nx; i < nx + n; i++) hm_texel(m, i, j);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, j & mask, n, 1, GL_RG, GL_FLOAT, m->data + 2 * (size_t)(j & mask) * n);
    }
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    /* the terrain samples mips (a triangle's worth of ground), so keep them current */
    if (nx != m->ox || nz != m->oz) glGenerateMipmap(GL_TEXTURE_2D);
    m->ox = nx; m->oz = nz;
}

/* Bilinear height/forest from the fine map's CPU copy (must cover x, z). */
static void hm0_sample(double x, double z, float *h, float *f) {
    HMap *m = &hm[0];
    double gx = x / BASE, gz = z / BASE, fx = floor(gx), fz = floor(gz);
    int32_t i = (int32_t)fx, j = (int32_t)fz, mk = m->n - 1;
    float tx = (float)(gx - fx), tz = (float)(gz - fz);
    const float *a = m->data + 2 * ((size_t)(j & mk) * m->n + (i & mk));
    const float *b = m->data + 2 * ((size_t)(j & mk) * m->n + ((i + 1) & mk));
    const float *c = m->data + 2 * ((size_t)((j + 1) & mk) * m->n + (i & mk));
    const float *d = m->data + 2 * ((size_t)((j + 1) & mk) * m->n + ((i + 1) & mk));
    for (int k = 0; k < 2; k++) {
        float v = (a[k] + (b[k] - a[k]) * tx) * (1 - tz) + (c[k] + (d[k] - c[k]) * tx) * tz;
        if (k == 0) *h = v; else *f = v;
    }
}

/* ---- GL helpers ---- */
#define GLSL(...) #__VA_ARGS__

static GLuint compile(GLenum type, const char *const *src, int n, const char *name) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, n, src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[8192];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "flightsim: %s %s shader:\n%s\n", name,
                type == GL_VERTEX_SHADER ? "vertex" : type == GL_FRAGMENT_SHADER ? "fragment" :
                type == GL_TESS_CONTROL_SHADER ? "tess control" : "tess evaluation", log);
        exit(1);
    }
    return s;
}

static const char *GLSL_HEAD = "#version 450 core\n";
static const char *GLSL_COMMON;

/* tcs/tes: optional tessellation stages (both or neither) */
/* fs_pre: extra fragment-shader source ahead of the common code (variants) */
static GLuint program_full(const char *name, const char *vs, const char *tcs, const char *tes, const char *fs, const char *fs_pre) {
    const char *v[3] = {GLSL_HEAD, GLSL_COMMON, vs}, *f[4] = {GLSL_HEAD, fs_pre, GLSL_COMMON, fs};
    GLuint p = glCreateProgram();
    glAttachShader(p, compile(GL_VERTEX_SHADER, v, 3, name));
    if (tcs) {
        const char *c[3] = {GLSL_HEAD, GLSL_COMMON, tcs}, *e[3] = {GLSL_HEAD, GLSL_COMMON, tes};
        glAttachShader(p, compile(GL_TESS_CONTROL_SHADER, c, 3, name));
        glAttachShader(p, compile(GL_TESS_EVALUATION_SHADER, e, 3, name));
    }
    glAttachShader(p, compile(GL_FRAGMENT_SHADER, f, 4, name));
    glLinkProgram(p);
    GLint ok;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[8192];
        glGetProgramInfoLog(p, sizeof log, NULL, log);
        fprintf(stderr, "flightsim: link %s:\n%s\n", name, log);
        exit(1);
    }
    glUseProgram(p);
    /* fixed texture units */
    const char *units[] = {"uH0", "uH1", "uShadow", "uNoise2", "uNoise3", "uSrc", "uBloom"};
    for (int i = 0; i < 7; i++) {
        GLint l = glGetUniformLocation(p, units[i]);
        if (l >= 0) glUniform1i(l, i);
    }
    GLint l = glGetUniformLocation(p, "uHist");                 /* unit 7 is the upload scratch unit */
    if (l < 0) l = glGetUniformLocation(p, "uCloudTex");
    if (l >= 0) glUniform1i(l, 8);
    if ((l = glGetUniformLocation(p, "uDepth")) >= 0) glUniform1i(l, 9);
    if ((l = glGetUniformLocation(p, "uShaft")) >= 0) glUniform1i(l, 10);
    if ((l = glGetUniformLocation(p, "uRefl")) >= 0) glUniform1i(l, 11);
    if ((l = glGetUniformLocation(p, "uReflD")) >= 0) glUniform1i(l, 12);
    if ((l = glGetUniformLocation(p, "uSSR")) >= 0) glUniform1i(l, 13);
    if ((l = glGetUniformLocation(p, "uWave")) >= 0) glUniform1i(l, 14);
    if ((l = glGetUniformLocation(p, "uDetail")) >= 0) glUniform1i(l, 15);
    return p;
}

static GLuint program_tess(const char *name, const char *vs, const char *tcs, const char *tes, const char *fs) {
    return program_full(name, vs, tcs, tes, fs, "");
}
static GLuint program(const char *name, const char *vs, const char *fs) { return program_tess(name, vs, NULL, NULL, fs); }

#define UL(p, n) glGetUniformLocation(p, n)

/* ---- shaders ---- */
static const char *COMMON_SRC = GLSL(
uniform vec3 uSunDir; uniform vec3 uSunCol; uniform vec3 uZen; uniform vec3 uHor; uniform vec3 uHorSun;
uniform vec3 uAmbSky; uniform vec3 uAmbGnd;
uniform float uCamY; uniform float uTime;
uniform float uFogA; uniform float uFogB; uniform float uHaze; uniform float uFogFar;
uniform float uMist;                 /* valley mist density inside the layer (1/m); 0 = none */
uniform float uMistTop0;             /* mistTop() under the camera */
uniform float uGlow; uniform float uStars; uniform float uDisc;
uniform vec2 uWorldMod; uniform vec2 uCloudMod;
uniform float uCloudBase; uniform float uCloudThick; uniform float uCloudCover; uniform float uCloudDens; uniform float uCloudsOn;
uniform sampler2D uH0; uniform sampler2D uH1; uniform sampler2D uShadow;
uniform sampler2D uNoise2; uniform sampler3D uNoise3;
uniform sampler2D uDetail;           /* ground detail: RG slope, B soft fbm, A ridged height */
uniform vec2 uT0Cam; uniform vec2 uT1Cam;
uniform float uPixAng;               /* radians per scene pixel */

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
/* Tiling textures without the tiling showing: each layer is turned by a
 * different angle whose matrix is integer, (a b; -b a), and scaled by a power
 * of two, so the world-coordinate wrap (uWorldMod, 16384 m) still lands on
 * whole tiles. The layer's tile is S / |(a, b)| metres. */
vec2 turn(vec2 w, vec2 m, float S) { return vec2(m.x * w.x + m.y * w.y, -m.y * w.x + m.x * w.y) / S; }
/* a slope read in turned texture space, back into world axes */
vec2 unturn(vec2 g, vec2 m) { return vec2(m.x * g.x - m.y * g.y, m.y * g.x + m.x * g.y) / length(m); }

/* interleaved gradient noise: a jitter that tiles smoothly, unlike white noise */
float ign(vec2 p) { return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }
float hash13(vec3 p3) {
    p3 = fract(p3 * .1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}

/* the sky without the sun disc: also the colour of the haze */
vec3 skyBase(vec3 rd) {
    float y = max(rd.y, 0.0);
    vec2 h = normalize(rd.xz + vec2(1e-5, 0.0)), s = normalize(uSunDir.xz + vec2(1e-5, 0.0));
    float toward = dot(h, s) * 0.5 + 0.5;
    vec3 hc = mix(uHor, uHorSun, pow(toward, 3.0) * (1.0 - smoothstep(0.0, 0.5, y)));
    vec3 col = mix(hc, uZen, 1.0 - exp(-y * 3.2));
    float mu = max(dot(rd, uSunDir), 0.0);
    col += uSunCol * uGlow * (0.018 * pow(mu, 5.0) + 0.09 * pow(mu, 48.0) + 0.5 * pow(mu, 900.0));
    return col;
}
vec3 sunDisc(vec3 rd) {
    float mu = dot(rd, uSunDir);
    return uSunCol * 40.0 * smoothstep(cos(uDisc), cos(uDisc * 0.8), mu);
}
vec3 stars(vec3 rd) {
    if (uStars <= 0.0 || rd.y < 0.0) return vec3(0.0);
    vec3 q = rd * 420.0, id = floor(q), f = fract(q) - 0.5;
    float h = hash13(id);
    if (h < 0.991) return vec3(0.0);
    float b = (h - 0.991) / 0.009;
    float tw = 0.75 + 0.25 * sin(uTime * (2.0 + 5.0 * hash13(id + 7.0)) + h * 60.0);
    return mix(vec3(1.0, 0.8, 0.6), vec3(0.7, 0.8, 1.0), hash13(id + 3.0)) *
           smoothstep(0.3, 0.0, length(f)) * b * b * 2.5 * tw * uStars * smoothstep(0.0, 0.2, rd.y);
}

/* weight of the fine heightmap (+-4 km around the camera) over the coarse one */
float fineW(float cheb) { return 1.0 - smoothstep(2800.0, 3600.0, cheb); }

/* transmittance of the height fog + haze over t metres along rd */
float fogTrans(vec3 rd, float t) {
    float k = rd.y * uFogB;
    float od = uFogA * exp(-uFogB * uCamY) * (abs(k) > 1e-5 ? (1.0 - exp(-k * t)) / k : t) + uHaze * t;
    return exp(-od) * (1.0 - smoothstep(uFogFar * 0.7, uFogFar, t));
}
vec3 fogColor(vec3 rd) { return skyBase(rd); }

/* Valley mist: a layer whose top sits a little under the ground's average
 * height over ~2 km (so it pools in whatever valleys there are, and over the
 * lakes), full density below the top and thinning fast above it. */
float mistTop(vec2 xz) {
    return textureLod(uH1, (uT1Cam + xz / 32.0 + 0.5) / 1024.0, 6.0).r - 80.0;
}
/* optical depth along rd for t metres: the height above the mist top runs
 * linearly from the camera's to the end's, so it integrates exactly */
float mistDepth(vec3 rd, float t) {
    if (uMist <= 0.0) return 0.0;
    vec3 e = rd * t;
    const float Hm = 20.0;
    float u0 = uCamY - uMistTop0, u1 = uCamY + e.y - mistTop(e.xz);
    /* antiderivative of the density profile (over height above the top) */
    vec2 F = vec2(u0, u1);
    F = mix(F, Hm * (1.0 - exp(-F / Hm)), step(0.0, F));
    float du = u1 - u0;
    float od = abs(du) > 1e-3 ? (F.y - F.x) / du : (u0 < 0.0 ? 1.0 : exp(-u0 / Hm));
    /* patchy, drifting slowly */
    vec2 w = uWorldMod + e.xz + vec2(uTime * 1.5, uTime * 0.6);
    float patchy = texture(uNoise2, w / 3000.0).g;
    /* none right round the camera: it's for looking down or across on, not
     * for flying blind through */
    return uMist * t * od * smoothstep(0.2, 0.8, patchy) * 1.6 * smoothstep(100.0, 600.0, t);
}
/* mist is whiter than the haze, lit a little by the sun */
vec3 mistColor(vec3 rd) {
    vec3 f = fogColor(rd);
    float l = dot(f, vec3(0.2126, 0.7152, 0.0722));
    return mix(f, vec3(l * 1.05), 0.45) + uSunCol * max(uSunDir.y, 0.0) * 0.02;
}
vec3 applyMist(vec3 col, vec3 rd, float t, float alpha) {
    float T = max(exp(-mistDepth(rd, t)), 0.25);      /* never a whiteout */
    return col * T + mistColor(rd) * (1.0 - T) * alpha;
}

/* cumulus: a 2D coverage field sets each column's height, 3D billow noise erodes it */
/* fp: the sample's footprint in metres. The raymarch picks mips itself: the
 * hardware's choice is garbage inside a loop that pixels leave at different
 * steps, and that shows as speckle. */
float cloudCovLod(vec2 xz, float fp) {
    vec2 w = uCloudMod + xz;
    /* no finer than ~128 m: this 2D field is extruded straight up into each
     * cloud's column, so any finer detail reads as vertical streaks */
    float n = textureLod(uNoise2, w / 8192.0, max(log2(fp / 32.0), 2.0)).r * 0.6 +
              textureLod(uNoise2, w / 2048.0, max(log2(fp / 8.0), 4.0)).g * 0.4;
    return smoothstep(1.0 - uCloudCover, 1.0 - uCloudCover + 0.28, n);
}
float cloudCov(vec2 xz) { return cloudCovLod(xz, 1.0); }
float cloudShadow(vec3 p) {
    if (uCloudsOn < 0.5) return 1.0;
    float y = uCloudBase + uCloudThick * 0.2;
    vec2 q = p.xz + uSunDir.xz / max(uSunDir.y, 0.06) * (y - p.y);
    /* soft-edged, like the penumbra of a thick cloud: a steep slope maps a few
     * metres of height onto a long run of the cloud field (up to 1/sin of the
     * sun's elevation), so any fine detail here comes out as stripes */
    float sy = max(uSunDir.y, 0.06);
    float fp = 400.0 * max(1.0, 0.25 / sy);          /* the blur grows with that squash */
    return 1.0 - 0.75 * mix(0.4, 1.0, smoothstep(0.05, 0.25, sy)) * smoothstep(0.03, 0.45, cloudCovLod(q, fp));
}
float hg(float mu, float g) { return (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * mu, 1.5); }

float gCloudT;                   /* clouds(): opacity-weighted distance of what it hit */
vec4 clouds(vec3 rd, float tmax, int steps, float jit) {
    gCloudT = 4000.0;
    if (uCloudsOn < 0.5) return vec4(0.0, 0.0, 0.0, 1.0);
    float top = uCloudBase + uCloudThick, t0, t1;
    if (abs(rd.y) < 1e-4) {
        if (uCamY < uCloudBase || uCamY > top) return vec4(0.0, 0.0, 0.0, 1.0);
        t0 = 0.0; t1 = tmax;
    } else {
        float a = (uCloudBase - uCamY) / rd.y, b = (top - uCamY) / rd.y;
        t0 = max(min(a, b), 0.0); t1 = min(max(a, b), tmax);
    }
    t1 = min(t1, uFogFar);
    if (t1 <= t0) return vec4(0.0, 0.0, 0.0, 1.0);
    float dt = (t1 - t0) / float(steps), T = 1.0, tw = 0.0, wsum = 0.0;
    gCloudT = 0.5 * (t0 + t1);
    vec3 L = vec3(0.0), fc = fogColor(rd);
    float mu = dot(rd, uSunDir);
    float phase = mix(hg(mu, -0.15), hg(mu, 0.7), 0.3);
    for (int i = 0; i < steps; i++) {
        float t = t0 + (float(i) + jit) * dt;
        vec3 p = vec3(0.0, uCamY, 0.0) + rd * t;
        /* footprint: a pixel's width out here (the jitter, averaged over
         * frames, takes care of the gaps between steps) */
        float fp = t * uPixAng;
        float cov = cloudCovLod(p.xz, fp);
        if (cov <= 0.0) continue;
        float h = (p.y - uCloudBase) / uCloudThick;
        float d = min(cov * 1.1 - h * h * 1.3 - 0.05, h * 2.0);
        if (d <= -0.3) continue;
        vec3 q = vec3(uCloudMod.x + p.x, p.y * 1.6 - uTime * 1.5, uCloudMod.y + p.z) / 1024.0;
        float lod = log2(fp / 16.0);
        float bil = textureLod(uNoise3, q, lod).r * 0.65 + textureLod(uNoise3, q * 3.7, lod + 1.89).r * 0.35;
        d -= (1.0 - bil) * (0.30 + 0.25 * (1.0 - cov));
        float dens = clamp(d * 3.0, 0.0, 1.0);
        if (dens <= 0.0) continue;
        /* light: optical depth up to the column top toward the sun, with a
         * long tail standing in for multiple scattering */
        float od = max(cov * 1.05 - h, 0.0) * uCloudThick * uCloudDens * 0.05 / max(uSunDir.y, 0.15);
        float sunT = exp(-od) * 0.7 + exp(-od * 0.15) * 0.3;
        float powder = 1.0 - exp(-dens * 3.0);
        vec3 amb = uAmbSky * mix(0.55, 1.35, h) + uAmbGnd * (1.0 - h) * 0.3;
        vec3 lum = uSunCol * sunT * phase * mix(0.6, 1.0, powder) * 0.9 + amb;
        float a = 1.0 - exp(-dens * uCloudDens * dt);
        float fT = fogTrans(rd, t);
        L += T * a * (lum * fT + fc * (1.0 - fT));
        tw += T * a * t; wsum += T * a;
        T *= 1.0 - a;
        if (T < 0.02) break;
    }
    if (wsum > 1e-3) gCloudT = tw / wsum;
    return vec4(L, T);
}

/* baked terrain sun shadow (r) and ambient occlusion (g) */
vec2 terrShadow(vec3 p) { return texture(uShadow, (uT1Cam + p.xz / 32.0 + 0.5) / 1024.0).rg; }

/* atmos() as col * a + rgb (for per-vertex use: small things like trees) */
vec4 atmosParts(vec3 rd, float t, int steps, float jit) {
    float T = fogTrans(rd, t);
    vec3 add = fogColor(rd) * (1.0 - T);
    float Tm = max(exp(-mistDepth(rd, t)), 0.25);
    add = add * Tm + mistColor(rd) * (1.0 - Tm);
    float mul = T * Tm;
    if (uCloudsOn > 0.5 && (uCamY > uCloudBase || rd.y > 0.0)) {
        vec4 c = clouds(rd, t, steps, jit);
        add = add * c.a + c.rgb;
        mul *= c.a;
    }
    return vec4(add, mul);
}
/* fog, then any cloud in front of a surface t metres away */
vec3 atmos(vec3 col, vec3 rd, float t, int steps, float jit) {
    float T = fogTrans(rd, t);
    col = col * T + fogColor(rd) * (1.0 - T);
    col = applyMist(col, rd, t, 1.0);
    if (uCloudsOn > 0.5 && (uCamY > uCloudBase || rd.y > 0.0)) {
        vec4 c = clouds(rd, t, steps, jit);
        col = col * c.a + c.rgb;
    }
    return col;
}
);

static const char *VS_FULL = GLSL(
out vec2 vUV;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    vUV = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
);

/* Sky clouds, accumulated over frames: each frame marches with a fresh jitter
 * and blends into last frame's result, reprojected to where the cloud it hit
 * was then (camera turn and travel both), so ~56 steps look like hundreds. */
static const char *FS_CLOUD = GLSL(
in vec2 vUV;
uniform vec4 uViewTan; uniform mat3 uCamRot;
uniform vec4 uPrevTan; uniform mat3 uPrevRot; uniform vec3 uPrevCam;   /* uPrevCam: last camera, relative to this one */
uniform sampler2D uHist; uniform float uHistW; uniform float uFrame;
out vec4 oCol;
void main() {
    vec3 rd = normalize(uCamRot * vec3(uViewTan.x + vUV.x * uViewTan.z, uViewTan.y + vUV.y * uViewTan.w, -1.0));
    vec4 c = clouds(rd, 1e9, 32, fract(ign(gl_FragCoord.xy) + uFrame * 0.618034));
    vec3 v = transpose(uPrevRot) * (rd * gCloudT - uPrevCam);
    if (uHistW > 0.0 && v.z < -1.0) {
        vec2 uv = (v.xy / -v.z - uPrevTan.xy) / uPrevTan.zw;
        if (all(greaterThanEqual(uv, vec2(0.0))) && all(lessThanEqual(uv, vec2(1.0))))
            c = mix(c, texture(uHist, uv), uHistW);
    }
    oCol = c;
}
);

static const char *FS_SKY = GLSL(
in vec2 vUV;
uniform vec4 uViewTan; uniform mat3 uCamRot;
uniform sampler2D uCloudTex;
out vec4 oCol;
void main() {
    vec3 rd = normalize(uCamRot * vec3(uViewTan.x + vUV.x * uViewTan.z, uViewTan.y + vUV.y * uViewTan.w, -1.0));
    vec3 col = skyBase(rd) + sunDisc(rd) + stars(rd);
    col = applyMist(col, rd, uFogFar, 1.0);
    vec4 c = uCloudsOn > 0.5 ? texture(uCloudTex, vUV) : vec4(0.0, 0.0, 0.0, 1.0);
    oCol = vec4(col * c.a + c.rgb, 1.0);
}
);

/* Terrain: a world-anchored grid of PATCH-metre patches, each tessellated on
 * the GPU so its triangles are about TESS_PX pixels wide wherever they are.
 * Heights come from the heightmap mip that matches a triangle's size, so the
 * ground only ever changes by sub-triangle detail as it comes closer, rather
 * than whole ridges growing in as fixed LOD rings pass over them. */
static const char *VS_TERRAIN = GLSL(
uniform ivec2 uPatch0; uniform int uPatchCells;
uniform ivec2 uCamCell; uniform vec2 uCamFrac;
/* patches nearest the camera first, so early depth rejects what they hide */
layout(std430, binding = 0) readonly buffer PatchOrder { ivec2 order[]; };
out vec2 vXZ;
void main() {
    ivec2 pi = uPatch0 + order[gl_InstanceID] + ivec2(gl_VertexID & 1, gl_VertexID >> 1);
    vXZ = vec2(pi * uPatchCells - uCamCell) * 8.0 - uCamFrac;
}
);

static const char *TCS_TERRAIN = GLSL(
layout(vertices = 4) out;
in vec2 vXZ[]; out vec2 tXZ[];
uniform mat4 uVP; uniform float uTessK;        /* uTessK: 1 / (radians per triangle) */
float hLod(vec2 xz, float lod) { return max(textureLod(uH1, (uT1Cam + xz / 32.0 + 0.5) / 1024.0, lod).r, 0.0); }
/* how far the ground strays from a straight line between a and b (m) */
float bumpiness(vec2 a, vec2 b) {
    float ha = hLod(a, 1.0), hb = hLod(b, 1.0), dev = 0.0;
    for (int i = 1; i < 8; i++) {
        float f = float(i) / 8.0;
        dev = max(dev, abs(hLod(mix(a, b, f), 1.0) - mix(ha, hb, f)));
    }
    return dev;
}
/* screen-sized triangles, fewer where the ground is smooth enough that
 * coarser ones lose nothing. Depends on the edge's two ends alone, so
 * neighbours agree and never crack. */
float level(vec2 a, vec2 b, float bump) {
    vec2 m = 0.5 * (a + b);
    float d = length(vec3(m.x, hLod(m, 3.0) - uCamY, m.y));
    return clamp(length(b - a) * uTessK / max(d, 1.0) * clamp(bump / 60.0, 0.25, 1.0), 1.0, 64.0);
}
float edgeLevel(vec2 a, vec2 b) { return level(a, b, bumpiness(a, b)); }
bool outside(vec4 c[8], int axis, float sgn) {
    for (int i = 0; i < 8; i++) if (sgn * c[i][axis] < c[i].w) return false;
    return true;
}
void main() {
    tXZ[gl_InvocationID] = vXZ[gl_InvocationID];
    if (gl_InvocationID != 0) return;
    /* cull: beyond the fog, or wholly off one side of the view */
    vec2 lo = min(min(vXZ[0], vXZ[1]), min(vXZ[2], vXZ[3])), hi = max(max(vXZ[0], vXZ[1]), max(vXZ[2], vXZ[3]));
    vec2 nearest = clamp(vec2(0.0), lo, hi);
    bool cull = length(nearest) > uFogFar;
    if (!cull) {
        vec4 c[8];
        for (int i = 0; i < 8; i++) c[i] = uVP * vec4(vXZ[i & 3].x, (i < 4 ? -120.0 : 2400.0), vXZ[i & 3].y, 1.0);
        cull = outside(c, 0, 1.0) || outside(c, 0, -1.0) || outside(c, 1, 1.0) || outside(c, 1, -1.0);
    }
    if (cull) {
        gl_TessLevelOuter[0] = gl_TessLevelOuter[1] = gl_TessLevelOuter[2] = gl_TessLevelOuter[3] = 0.0;
        gl_TessLevelInner[0] = gl_TessLevelInner[1] = 0.0;
        return;
    }
    /* corners: 0 (0,0), 1 (1,0), 2 (0,1), 3 (1,1) */
    gl_TessLevelOuter[0] = edgeLevel(vXZ[0], vXZ[2]);
    gl_TessLevelOuter[1] = edgeLevel(vXZ[0], vXZ[1]);
    gl_TessLevelOuter[2] = edgeLevel(vXZ[1], vXZ[3]);
    gl_TessLevelOuter[3] = edgeLevel(vXZ[2], vXZ[3]);
    /* inside: its own bumpiness (the midlines and diagonals); only the
     * edges have to match the neighbours */
    vec2 m01 = 0.5 * (vXZ[0] + vXZ[1]), m23 = 0.5 * (vXZ[2] + vXZ[3]), m02 = 0.5 * (vXZ[0] + vXZ[2]), m13 = 0.5 * (vXZ[1] + vXZ[3]);
    float bi = max(max(bumpiness(m01, m23), bumpiness(m02, m13)), max(bumpiness(vXZ[0], vXZ[3]), bumpiness(vXZ[1], vXZ[2])));
    gl_TessLevelInner[0] = max(level(m02, m13, bi), max(gl_TessLevelOuter[1], gl_TessLevelOuter[3]) * 0.5);
    gl_TessLevelInner[1] = max(level(m01, m23, bi), max(gl_TessLevelOuter[0], gl_TessLevelOuter[2]) * 0.5);
}
);

static const char *TES_TERRAIN = GLSL(
layout(quads, fractional_even_spacing, ccw) in;
in vec2 tXZ[];
uniform mat4 uVP; uniform float uTessK;
out vec3 vRel;
void main() {
    vec2 u = gl_TessCoord.xy;
    vec2 xz = mix(mix(tXZ[0], tXZ[1], u.x), mix(tXZ[2], tXZ[3], u.x), u.y);
    /* the mip whose texels are about a triangle wide here (a function of the
     * position only, so shared edges get the same heights) */
    float lod = log2(max(length(xz) / uTessK, 8.0) / 8.0);
    float h = textureLod(uH1, (uT1Cam + xz / 32.0 + 0.5) / 1024.0, max(lod - 2.0, 0.0)).r;
    float w0 = fineW(max(abs(xz.x), abs(xz.y)));
    if (w0 > 0.0) h = mix(h, textureLod(uH0, (uT0Cam + xz / 8.0 + 0.5) / 1024.0, lod).r, w0);
    vRel = vec3(xz.x, h, xz.y);
    gl_Position = uVP * vec4(vRel, 1.0);
}
);

static const char *FS_TERRAIN = GLSL(
in vec3 vRel;
out vec4 oCol;
/* Normal from the heightmap by central differences. The mip is picked from
 * the distance (texel size e metres), not the hardware's screen derivatives:
 * those jump between mips across tiny triangles and draw bands on the
 * hills. The step grows with the mip to match. */
float gDist;
vec3 hmNormal(sampler2D tex, vec2 uv, float px, float e, out float forest) {
    float lod = max(log2(gDist * uPixAng * 1.5 / e), 0.0), k = exp2(lod);
    vec2 c = textureLod(tex, uv, lod).rg;
    forest = c.g;
    float hl = textureLod(tex, uv - vec2(px * k, 0.0), lod).r, hr = textureLod(tex, uv + vec2(px * k, 0.0), lod).r;
    float hd = textureLod(tex, uv - vec2(0.0, px * k), lod).r, hu = textureLod(tex, uv + vec2(0.0, px * k), lod).r;
    return normalize(vec3(hl - hr, 2.0 * e * k, hd - hu));
}
float nz(vec2 w, vec2 m, float S, int ch) { vec4 v = texture(uNoise2, turn(w, m, S)); return ch == 0 ? v.r : ch == 1 ? v.g : ch == 2 ? v.b : v.a; }
void main() {
    vec3 p = vRel, cam = vec3(0.0, uCamY, 0.0);
    float dist = length(p - cam);
    gDist = dist;
    vec3 rd = (p - cam) / dist;
    float cheb = max(abs(p.x), abs(p.z));
    float f1, f0, forest;
    float w0 = fineW(cheb);
    vec3 n = vec3(0.0, 1.0, 0.0);
    if (w0 < 1.0) n = hmNormal(uH1, (uT1Cam + p.xz / 32.0 + 0.5) / 1024.0, 1.0 / 1024.0, 32.0, f1);
    if (w0 > 0.0) {
        vec3 n0 = hmNormal(uH0, (uT0Cam + p.xz / 8.0 + 0.5) / 1024.0, 1.0 / 1024.0, 8.0, f0);
        n = w0 < 1.0 ? normalize(mix(n, n0, w0)) : n0;
        forest = w0 < 1.0 ? mix(f1, f0, w0) : f0;
    } else forest = f1;
    vec2 w = uWorldMod + p.xz;
    /* ~1.6 km, ~315 m and ~30 m, each turned differently */
    float n1 = nz(w, vec2(3.0, 4.0), 8192.0, 0), n2 = nz(w, vec2(5.0, 12.0), 4096.0, 1);
    /* the fine one twice, turned apart, so its lattice doesn't read as a grid */
    float n3 = clamp(0.5 + 1.35 * (0.5 * (nz(w, vec2(8.0, -15.0), 512.0, 2) + nz(w, vec2(12.0, 5.0), 256.0, 2)) - 0.5), 0.0, 1.0);

    /* small-scale relief the heightmap can't carry: crisp ridged detail at
     * 128, 32 and 8 m, projected from above and from the side the slope faces
     * most (so cliffs don't smear), fading out before it would shimmer */
    float cav = 0.5, cav2 = 0.5;
    float far = 1.0 - smoothstep(2000.0, 9000.0, dist);
    float near = 1.0 - smoothstep(300.0, 2500.0, dist), fine = 1.0 - smoothstep(60.0, 600.0, dist);
    if (far > 0.0) {
        float rockiness = smoothstep(0.12, 0.4, 1.0 - n.y);
        vec3 ww = vec3(w.x, p.y, w.y);
        vec3 bw = pow(abs(n), vec3(4.0));
        float top = bw.y / (bw.x + bw.y + bw.z);
        bool sx = bw.x > bw.z;
        vec2 sideUV = sx ? ww.zy : ww.xy;
        vec3 dn = vec3(0.0);
        for (int k = 0; k < 3; k++) {
            /* ~102, ~39 and ~7.5 m tiles, turned apart so they never line up */
            vec2 m = k == 0 ? vec2(3.0, 4.0) : k == 1 ? vec2(5.0, 12.0) : vec2(8.0, -15.0);
            float S = k == 2 ? 128.0 : 512.0, amt = k == 0 ? far : k == 1 ? near : fine;
            if (amt <= 0.0) break;
            /* skip a projection that hardly shows (most ground is flat-ish) */
            /* (the 128 m layer is seen from afar: its dominant one will do) */
            float lo = k == 0 ? 0.5 : 0.03;
            vec4 t0 = top > lo ? texture(uDetail, turn(ww.xz, m, S)) : vec4(0.5);
            vec4 t1 = top < 1.0 - lo ? texture(uDetail, turn(sideUV, m, S)) : vec4(0.5);
            vec2 g0 = unturn(t0.rg * 2.0 - 1.0, m), g1 = unturn(t1.rg * 2.0 - 1.0, m);
            /* a slope in a projection plane tilts the normal against it */
            vec3 d = vec3(g0.x, 0.0, g0.y) * top;
            d += (sx ? vec3(0.0, g1.y, g1.x) : vec3(g1.x, g1.y, 0.0)) * (1.0 - top);
            /* rougher and smoother ground by region, so no two hills match */
            float str = (k == 0 ? 0.8 : k == 1 ? 1.1 : 0.55) * amt * mix(0.12, 1.0, rockiness) * (0.45 + 1.1 * n2);
            dn += d * str;
            float c = mix(t0.a, t1.a, 1.0 - top);
            if (k == 1) cav = mix(0.5, c, amt); else if (k == 2) cav2 = mix(0.5, c, amt);
        }
        n = normalize(n - dn);
    }
    float slope = 1.0 - n.y;
    float h = p.y, hN = h + (n2 - 0.5) * 60.0;

    /* materials (linear albedo) */
    vec3 grassA = vec3(0.07, 0.13, 0.035), grassB = vec3(0.15, 0.16, 0.07), alpine = vec3(0.13, 0.15, 0.07);
    vec3 canopy = vec3(0.028, 0.065, 0.025), sand = vec3(0.42, 0.36, 0.24), wet = vec3(0.20, 0.17, 0.12);
    vec3 rock = vec3(0.19, 0.165, 0.14), snow = vec3(0.86, 0.89, 0.94);
    vec3 col = mix(grassA, grassB, smoothstep(0.35, 0.75, n1 * 0.7 + n3 * 0.3));
    col = mix(col, alpine, smoothstep(700.0, 1000.0, hN));
    col *= (0.85 + 0.3 * n3) * (0.88 + 0.24 * cav2);           /* tufts */
    float fo = smoothstep(0.15, 0.55, forest) * (1.0 - smoothstep(0.35, 0.5, slope));
    col = mix(col, canopy * (0.8 + 0.4 * n3), fo);          /* (a finer noise here drew a lattice on wooded hills) */
    col = mix(sand, col, smoothstep(0.5, 3.5, h - 2.0 * n3));
    float strata = 0.9 + 0.1 * sin(h * 0.05 + n2 * 6.0) + 0.25 * (n3 - 0.5);
    /* rock: crevices dark, ridges and ledges light */
    vec3 rc = rock * strata * mix(vec3(1.0), vec3(1.15, 0.95, 0.8), n1) * (0.55 + 0.6 * cav) * (0.8 + 0.4 * cav2);
    float rockAmt = smoothstep(0.26, 0.40, slope + (n2 - 0.5) * 0.18) + smoothstep(1250.0, 1500.0, hN) * 0.6;
    col = mix(col, rc, clamp(rockAmt, 0.0, 1.0));
    float snowline = 1080.0 + (n1 - 0.5) * 260.0;
    float snowAmt = smoothstep(snowline - 40.0, snowline + 60.0, hN) * smoothstep(0.42, 0.22, slope + (n3 - 0.5) * 0.15);
    col = mix(col, snow, snowAmt);
    if (h < 0.5) col = mix(wet, sand * 0.6, smoothstep(0.0, -8.0, h));

    /* lighting */
    vec2 sa = terrShadow(p);
    float sh = sa.r * cloudShadow(p);
    float diff = max(dot(n, uSunDir), 0.0);
    vec3 amb = mix(uAmbGnd, uAmbSky, n.y * 0.5 + 0.5) * (0.35 + 0.65 * sa.g) * (1.0 - 0.3 * fo);
    vec3 lit = col * (uSunCol * diff * sh + amb);
    vec3 H = normalize(uSunDir - rd);
    lit += uSunCol * snowAmt * sh * pow(max(dot(n, H), 0.0), 40.0) * 0.25;
    oCol = vec4(atmos(lit, rd, dist, 10, ign(gl_FragCoord.xy)), 1.0);
}
);

static const char *VS_WATER = GLSL(
layout(location = 0) in vec2 aP;
uniform mat4 uVP; uniform float uExtent;
out vec3 vRel;
void main() {
    vRel = vec3(aP.x * uExtent, 0.0, aP.y * uExtent);
    gl_Position = uVP * vec4(vRel, 1.0);
}
);

static const char *FS_WATER = GLSL(
in vec3 vRel;
out vec4 oCol;
uniform sampler2D uSSR;        /* reflections off flat water, quarter res (FS_SSR) */
uniform vec2 uSceneSize;
uniform sampler2D uWave;
/* slope of one scrolling, turned layer of the wave map (see turn()) */
vec2 waveSlope(vec2 w, vec2 m, float S, float speed) {
    vec2 g = texture(uWave, turn(w, m, S) + vec2(uTime * speed * length(m) / S, 0.0)).rg * 2.0 - 1.0;
    return unturn(g, m);
}
void main() {
    vec3 p = vRel, cam = vec3(0.0, uCamY, 0.0);
    float cheb = max(abs(p.x), abs(p.z));
    float hb = texture(uH1, (uT1Cam + p.xz / 32.0 + 0.5) / 1024.0).r;
    float w0 = fineW(cheb);
    if (w0 > 0.0) hb = mix(hb, texture(uH0, (uT0Cam + p.xz / 8.0 + 0.5) / 1024.0).r, w0);
    float depth = -hb;
    if (depth <= 0.0) discard;
    float dist = length(p - cam);
    vec3 rd = (p - cam) / dist;
    vec2 w = uWorldMod + p.xz;

    /* waves: two layers of the wave map, crossing, drifting with the wind.
     * Gusts ruffle some patches while others lie glassy (they show the
     * reflections best), and the shallows are calmer. The map's mips
     * smooth the slopes with distance. */
    float gust = smoothstep(0.3, 0.7, texture(uNoise2, (w + vec2(uTime * 4.0, uTime * 1.5)) / 2048.0).g);
    float amp = mix(0.3, 1.0, gust) * mix(0.5, 1.0, smoothstep(0.5, 5.0, depth));
    /* ~51 m and ~20 m tiles */
    vec2 g = waveSlope(w, vec2(4.0, 3.0), 256.0, 1.6) * 0.22 + waveSlope(w, vec2(12.0, -5.0), 256.0, 0.8) * 0.12;
    vec3 n = normalize(vec3(-g.x * amp, 1.0, -g.y * amp));
    vec3 R = reflect(rd, n);
    R.y = abs(R.y);
    float F = 0.02 + 0.98 * pow(1.0 - max(dot(-rd, n), 0.0), 5.0);
    vec3 refl = skyBase(R);
    if (uCloudsOn > 0.5) {
        vec2 q = p.xz + R.xz / max(R.y, 0.03) * (uCloudBase + uCloudThick * 0.3);
        float c = smoothstep(0.05, 0.45, cloudCov(q)) * smoothstep(0.0, 0.08, R.y);
        refl = mix(refl, uAmbSky * 1.4 + uSunCol * 0.25 * max(uSunDir.y, 0.1), c * 0.85);
    }
    /* mountains and trees on screen reflect; the sky above stays the fallback */
    /* traced for flat water; the waves wobble where it's read */
    vec4 sr = texture(uSSR, gl_FragCoord.xy / uSceneSize + n.xz * vec2(0.03, -0.03));
    refl = mix(refl, sr.rgb, sr.a);
    float sh = terrShadow(p).r * cloudShadow(p);
    vec3 H = normalize(uSunDir - rd);
    float nh = max(dot(n, H), 0.0);
    /* the glint widens with distance (many waves per pixel) instead of
     * breaking into sparkles */
    float pw = mix(900.0, 90.0, smoothstep(40.0, 2500.0, dist));
    vec3 spec = uSunCol * sh * (pow(nh, pw) * pw * 0.045 + pow(nh, 60.0) * 0.8) * F * 4.0;
    vec3 body = mix(vec3(0.015, 0.09, 0.085), vec3(0.003, 0.018, 0.03), 1.0 - exp(-depth / 20.0));
    body *= uAmbSky * 1.5 + uSunCol * max(uSunDir.y, 0.0) * 0.35 * sh;
    float aBody = 1.0 - exp(-depth / 3.5);
    float fn = texture(uNoise2, w / 9.0 + vec2(uTime * 0.02, 0.0)).a;
    float foam = smoothstep(1.4, 0.2, depth + (fn - 0.5) * 1.6) * (0.55 + 0.45 * sin(uTime * 1.7 - depth * 5.0 + fn * 8.0));
    foam *= 1.0 - smoothstep(600.0, 2500.0, dist);
    vec3 col = F * refl + (1.0 - F) * body * aBody + spec;
    float alpha = clamp(F + (1.0 - F) * aBody, 0.0, 1.0);
    col = mix(col, (uSunCol * max(uSunDir.y, 0.0) * sh + uAmbSky) * 0.7, foam * 0.8);
    alpha = mix(alpha, 1.0, foam * 0.8);
    float T = fogTrans(rd, dist);
    col = col * T + fogColor(rd) * (1.0 - T) * alpha;
    col = applyMist(col, rd, dist, alpha);
    oCol = vec4(col, alpha);
}
);

static const char *VS_TREE = GLSL(
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNrm;
layout(location = 2) in float aAO;
layout(location = 3) in vec4 iA;   /* rel x, y, rel z, height */
layout(location = 4) in vec4 iB;   /* width, tint, rotation, type */
uniform mat4 uVP; uniform float uTreeR; uniform float uTreeNear; uniform float uTreeBand;
out float vLod;
out vec3 vRel; out vec3 vN; out float vAO; out float vFade; out float vTint; out float vShadow; out float vType;
out vec3 vObj;     /* metres in the tree's own frame, offset per tree: where its leaf clumps are */
out vec4 vAtm;     /* fog, mist and haze per vertex: col * a + rgb */
void main() {
    float c = cos(iB.z), s = sin(iB.z);
    vec3 lp = vec3(aPos.x * c - aPos.z * s, aPos.y, aPos.x * s + aPos.z * c);
    vec3 ln = vec3(aNrm.x * c - aNrm.z * s, aNrm.y, aNrm.x * s + aNrm.z * c);
    /* bend the foliage normals toward "out from the crown's middle", so the
     * crown lights as one soft round mass instead of facets */
    if (aAO >= 0.0) {
        vec3 ctr = iB.w < 0.5 ? vec3(0.0, max(aPos.y - 0.25, 0.3), 0.0) : vec3(0.0, 0.62, 0.0);
        vec3 r = aPos - ctr;
        vec3 rn = normalize(vec3(r.x * c - r.z * s, r.y, r.x * s + r.z * c));
        ln = normalize(mix(ln, rn, 0.45));
    }
    float sway = sin(uTime * 1.1 + iA.x * 0.05 + iA.z * 0.07) * 0.015 * aPos.y * aPos.y;
    vRel = vec3(iA.x + (lp.x + sway) * iB.x, iA.y + lp.y * iA.w, iA.z + lp.z * iB.x);
    vN = normalize(vec3(ln.x / iB.x, ln.y / iA.w, ln.z / iB.x));
    vObj = vec3(aPos.x * iB.x, aPos.y * iA.w, aPos.z * iB.x) + iB.y * 97.0 + iB.z * 13.0;
    vAO = aAO; vTint = iB.y; vType = iB.w;
    vFade = 1.0 - smoothstep(uTreeR * 0.72, uTreeR, length(iA.xz));   /* FADE0 */
    vLod = smoothstep(uTreeNear - uTreeBand, uTreeNear + uTreeBand, length(iA.xz));
    vShadow = terrShadow(iA.xyz).r * cloudShadow(iA.xyz);
    vec3 d = vRel - vec3(0.0, uCamY, 0.0);
    float dl = length(d);
    vAtm = atmosParts(d / dl, dl, 4, 0.5);
    gl_Position = uVP * vec4(vRel, 1.0);
}
);

static const char *FS_TREE = GLSL(
in vec3 vRel; in vec3 vN; in float vAO; in float vFade; in float vTint; in float vShadow; in float vType;
in float vLod; in vec3 vObj; in vec4 vAtm;
uniform int uFar;                /* drawing the far mesh: complementary dither across the LOD band */
out vec4 oCol;
void main() {
    /* leaf clumps: billowy 3D noise in the tree's frame (sampled before any
     * discard, so the mip choice is sound); pines get finer, flatter tufts */
    vec3 q = (vType < 0.5 ? vObj * vec3(0.55, 0.9, 0.55) : vObj * 0.42) / 5.0;
    float clump = texture(uNoise3, q).r;
    /* the same a little toward the sun: a clump is lit on its sunny side */
    float clumpL = texture(uNoise3, q + uSunDir * 0.04).r;
    if (kDither) {
        if (hash12(gl_FragCoord.xy) > vFade) discard;
        if ((hash12(gl_FragCoord.xy + 17.0) < vLod) != (uFar == 1)) discard;
    }
    vec3 n = normalize(vN), cam = vec3(0.0, uCamY, 0.0);
    float dist = length(vRel - cam);
    vec3 rd = (vRel - cam) / dist;
    vec3 base;
    bool trunk = vAO < 0.0;
    if (trunk) base = vec3(0.07, 0.045, 0.03);
    else if (vType < 0.5) base = mix(vec3(0.022, 0.058, 0.032), vec3(0.04, 0.078, 0.04), vTint);
    else {
        base = mix(vec3(0.045, 0.105, 0.028), vec3(0.085, 0.125, 0.032), smoothstep(0.0, 0.85, vTint));
        base = mix(base, vec3(0.13, 0.12, 0.035), smoothstep(0.93, 1.0, vTint));    /* the odd yellowing one */
    }
    float ao = abs(vAO);
    /* gaps between clumps: darker, and shadowed from the sun */
    /* clumps only up close: further out they'd just be noise */
    float cn = trunk ? 0.0 : 1.0 - smoothstep(60.0, 350.0, dist);
    float cl = mix(0.7, smoothstep(0.2, 0.8, clump), cn);
    base *= 0.8 + 0.35 * cl;
    float self = mix(1.0, clamp(0.65 + (clump - clumpL) * 5.0, 0.35, 1.15) * mix(0.8, 1.0, cl), cn);
    float diff = max(dot(n, uSunDir) * 0.8 + 0.2, 0.0) * self;
    /* sun through the leaves at the crown's edges when it's behind the tree */
    float rim = 1.0 - abs(dot(n, rd));
    float back = pow(max(dot(rd, uSunDir), 0.0), 4.0) * (0.35 + 0.9 * rim * rim) * (trunk ? 0.0 : 1.0);
    vec3 light = uSunCol * (diff + back * vec3(0.8, 1.3, 0.5)) * vShadow * mix(0.55, 1.0, ao) +
                 mix(uAmbGnd, uAmbSky, n.y * 0.5 + 0.5) * ao * mix(0.6, 1.0, cl);
    oCol = vec4(base * light * vAtm.a + vAtm.rgb, 1.0);
}
);

/* Birds: a 16-triangle bird whose wings bend at the shoulder and elbow.
 * aP.w: 0 body, 1 inner wing (shoulder..elbow), 2 outer wing (elbow..tip). */
static const char *VS_BIRD = GLSL(
layout(location = 0) in vec4 aP;
layout(location = 1) in vec4 iA;   /* rel x, y, z, heading (radians, 0 = -z) */
layout(location = 2) in vec4 iB;   /* flap angle, roll, scale, pitch */
uniform mat4 uVP;
out vec3 vRel; out vec3 vN; out vec4 vAtm;
void main() {
    vec3 p = aP.xyz;
    float a = iB.x, s = sign(p.x), ax = abs(p.x);
    vec3 n = vec3(0.0, 1.0, 0.0);
    if (aP.w > 0.5) {
        float r1 = min(ax, 0.45) - 0.08, r2 = max(ax - 0.45, 0.0);
        float a2 = aP.w > 1.5 ? a * 1.5 : a;
        p.x = s * (0.08 + r1 * cos(a) + r2 * cos(a2));
        p.y = r1 * sin(a) + r2 * sin(a2);
        n = vec3(-s * sin(a2), cos(a2), 0.0);
    }
    /* roll, pitch, then heading */
    float cr = cos(iB.y), sr = sin(iB.y), cp = cos(iB.w), sp = sin(iB.w), ch = cos(iA.w), sh = sin(iA.w);
    p = vec3(p.x * cr - p.y * sr, p.x * sr + p.y * cr, p.z);
    n = vec3(n.x * cr - n.y * sr, n.x * sr + n.y * cr, n.z);
    p = vec3(p.x, p.y * cp + p.z * sp, -p.y * sp + p.z * cp);
    n = vec3(n.x, n.y * cp + n.z * sp, -n.y * sp + n.z * cp);
    /* bird +z (forward) maps to the heading direction (-sin h, -cos h) */
    p = vec3(p.x * ch - p.z * sh, p.y, -p.x * sh - p.z * ch);
    n = vec3(n.x * ch - n.z * sh, n.y, -n.x * sh - n.z * ch);
    vRel = iA.xyz + p * iB.z;
    vN = n;
    vec3 d = vRel - vec3(0.0, uCamY, 0.0);
    float dl = length(d);
    vAtm = atmosParts(d / dl, dl, 4, 0.5);
    gl_Position = uVP * vec4(vRel, 1.0);
}
);

static const char *FS_BIRD = GLSL(
in vec3 vRel; in vec3 vN; in vec4 vAtm;
out vec4 oCol;
void main() {
    vec3 cam = vec3(0.0, uCamY, 0.0);
    float dist = length(vRel - cam);
    vec3 rd = (vRel - cam) / dist;
    vec3 n = normalize(vN);
    if (dot(n, rd) > 0.0) n = -n;                  /* two-sided */
    vec3 base = vec3(0.05, 0.045, 0.04);
    float diff = max(dot(n, uSunDir), 0.0);
    vec3 light = uSunCol * diff * cloudShadow(vRel) + mix(uAmbGnd, uAmbSky, n.y * 0.5 + 0.5);
    oCol = vec4(base * light * vAtm.a + vAtm.rgb, 1.0);
}
);

/* Bake the sun shadow + AO for every texel of the coarse map (in stripes). */
static const char *FS_BAKE = GLSL(
uniform ivec2 uOrg1;
out vec4 oCol;
float hAt(vec2 q) { return max(texture(uH1, (mod(q, 1024.0) + 0.5) / 1024.0).r, 0.0); }
void main() {
    ivec2 tx = ivec2(gl_FragCoord.xy);
    ivec2 wl = uOrg1 + ((tx - (uOrg1 & 1023)) & 1023);
    vec2 w = vec2(wl), lo = vec2(uOrg1), hi = vec2(uOrg1 + 1023);
    float h0 = max(texelFetch(uH1, tx, 0).r, 0.0) + 2.0;
    float hl = length(uSunDir.xz);
    vec2 dir = uSunDir.xz / max(hl, 1e-4);
    float rise = uSunDir.y / max(hl, 1e-4);
    float sh = 1.0, d = 0.6;
    for (int i = 0; i < 52; i++) {
        d += 0.5 + d * 0.07;
        vec2 q = w + dir * d;
        if (any(lessThan(q, lo)) || any(greaterThan(q, hi))) break;
        float s = (h0 + rise * d * 32.0 - hAt(q)) / (d * 32.0);
        sh = min(sh, clamp(0.5 + s * 14.0, 0.0, 1.0));
        if (sh <= 0.0) break;
    }
    float occ = 0.0;
    for (int k = 0; k < 10; k++) {
        float a = float(k) * 0.6283;
        vec2 dv = vec2(cos(a), sin(a));
        float m = 0.0;
        for (int r = 0; r < 3; r++) {
            float rr = r == 0 ? 1.5 : r == 1 ? 4.0 : 10.0;
            vec2 q = clamp(w + dv * rr, lo, hi);
            m = max(m, (hAt(q) - h0) / (rr * 32.0));
        }
        occ += m / sqrt(1.0 + m * m);
    }
    float ao = clamp(1.0 - occ / 10.0 * 1.6, 0.0, 1.0);
    oCol = vec4(sh, ao, 0.0, 1.0);
}
);

static const char *FS_DOWN = GLSL(
in vec2 vUV;
uniform sampler2D uSrc; uniform vec2 uTexel; uniform float uPrefilter;
out vec4 oCol;
void main() {
    vec2 o = uTexel;
    vec3 c = texture(uSrc, vUV).rgb * 4.0 + texture(uSrc, vUV + vec2(-o.x, -o.y)).rgb + texture(uSrc, vUV + vec2(o.x, -o.y)).rgb +
             texture(uSrc, vUV + vec2(-o.x, o.y)).rgb + texture(uSrc, vUV + vec2(o.x, o.y)).rgb;
    c /= 8.0;
    if (uPrefilter > 0.0) {
        c = min(c, vec3(80.0));
        float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
        c *= max(l - uPrefilter, 0.0) / max(l, 1e-4);
    }
    oCol = vec4(c, 1.0);
}
);

static const char *FS_UP = GLSL(
in vec2 vUV;
uniform sampler2D uSrc; uniform vec2 uTexel;
out vec4 oCol;
void main() {
    vec2 o = uTexel;
    vec3 c = texture(uSrc, vUV + vec2(-o.x * 2.0, 0.0)).rgb + texture(uSrc, vUV + vec2(o.x * 2.0, 0.0)).rgb +
             texture(uSrc, vUV + vec2(0.0, -o.y * 2.0)).rgb + texture(uSrc, vUV + vec2(0.0, o.y * 2.0)).rgb +
             (texture(uSrc, vUV + vec2(-o.x, -o.y)).rgb + texture(uSrc, vUV + vec2(o.x, -o.y)).rgb +
              texture(uSrc, vUV + vec2(-o.x, o.y)).rgb + texture(uSrc, vUV + vec2(o.x, o.y)).rgb) * 2.0;
    oCol = vec4(c / 12.0, 1.0);
}
);

static const char *FS_POST = GLSL(
in vec2 vUV;
uniform sampler2D uSrc; uniform sampler2D uBloom; uniform sampler2D uShaft; uniform float uShaftAmt;
uniform vec2 uTexel; uniform float uExposure; uniform float uBloomAmt; uniform float uSharpen; uniform float uSat;
uniform vec4 uVig;   /* this view in layout-normalised coords: x0, y0, w, h */
out vec4 oCol;
vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
void main() {
    vec3 c = texture(uSrc, vUV).rgb;
    if (uSharpen > 0.0) {
        /* unsharp mask against a local blur: two bilinear taps half a texel
         * off diagonally average the 3x3 neighbourhood */
        vec3 blur = 0.5 * (texture(uSrc, vUV + 0.5 * uTexel).rgb + texture(uSrc, vUV - 0.5 * uTexel).rgb);
        c = max(c + (c - blur) * 1.6 * uSharpen, vec3(0.0));
    }
    c += texture(uBloom, vUV).rgb * uBloomAmt;
    if (uShaftAmt > 0.0) c += texture(uShaft, vUV).rgb * uShaftAmt;
    c = aces(c * uExposure);
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c = max(mix(vec3(l), c, uSat), 0.0);
    vec2 q = uVig.xy + vUV * uVig.zw - 0.5;
    c *= 1.0 - 0.28 * smoothstep(0.25, 0.75, length(q * vec2(1.0, 0.8)));
    c = pow(c, vec3(1.0 / 2.2));
    c += (hash12(gl_FragCoord.xy) - 0.5) / 255.0;
    oCol = vec4(c, 1.0);
}
);

/* The scene so far, at half res, with its view depth in alpha, for the
 * water's reflections (read from a copy: the water draws into the scene) */
static const char *FS_REFLSRC = GLSL(
in vec2 vUV;
uniform sampler2D uSrc; uniform sampler2D uDepth;
layout(location = 0) out vec4 oCol;
layout(location = 1) out float oDepth;
void main() {
    float d = textureLod(uDepth, vUV, 0.0).r;
    oCol = vec4(textureLod(uSrc, vUV, 0.0).rgb, 1.0);
    oDepth = d > 0.0 ? min(1.5 / d, 1e5) : 1e5;         /* reversed-Z: near plane 1.5 */
}
);

/* Reflections of the scene in the water, traced for a flat surface at
 * quarter res: each pixel's ray meets the water plane, and if water shows there the
 * reflected ray is marched through the scene's depth. */
static const char *FS_SSR = GLSL(
in vec2 vUV;
uniform vec4 uViewTan; uniform mat3 uCamRot; uniform mat4 uVP;
uniform sampler2D uRefl;       /* the scene so far, half res */
uniform sampler2D uReflD;      /* its view depth, half res */
/* Screen-space reflection: march the reflected ray, projecting as it goes
 * (clip position is linear in t), until it passes behind what the scene drew
 * there. rgb, a = confidence. */
vec4 ssr(vec3 p, vec3 R) {
    vec4 c0 = uVP * vec4(p, 1.0), cr = uVP * vec4(R, 0.0);
    float t = 8.0, prev = 0.0;
    for (int i = 0; i < 15; i++) {
        vec4 c = c0 + cr * t;
        if (c.w < 1.0) break;
        vec2 uv = c.xy / c.w * 0.5 + 0.5;
        if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) break;
        float sd = textureLod(uReflD, uv, 0.0).r;
        if (c.w > sd) {
            /* crossed behind the scene somewhere in this step: find where,
             * then it's a hit if that's the surface and not a gap behind it */
            float a = prev, b = t, gap = c.w - sd;
            for (int k = 0; k < 7; k++) {
                float m = 0.5 * (a + b);
                vec4 cm = c0 + cr * m;
                vec2 um = cm.xy / cm.w * 0.5 + 0.5;
                float dm = textureLod(uReflD, um, 0.0).r;
                if (cm.w > dm) { b = m; uv = um; gap = cm.w - dm; } else a = m;
            }
            if (gap > max(30.0, 0.05 * t)) break;
            float edge = smoothstep(0.0, 0.1, min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y)));
            return vec4(textureLod(uRefl, uv, 0.0).rgb, edge);
        }
        prev = t;
        t *= 1.6;
    }
    return vec4(0.0);
}
out vec4 oCol;
void main() {
    oCol = vec4(0.0);
    vec3 v = vec3(uViewTan.x + vUV.x * uViewTan.z, uViewTan.y + vUV.y * uViewTan.w, -1.0);
    vec3 rd = normalize(uCamRot * v);
    if (rd.y > -0.002) return;
    float t = uCamY / -rd.y;
    /* view depth of the water there vs what's drawn: land in front hides it */
    if (t / length(v) > textureLod(uReflD, vUV, 0.0).r) return;
    vec3 p = rd * t;
    p.y = 0.0;
    vec3 R = reflect(rd, vec3(0.0, 1.0, 0.0));
    float F = 0.02 + 0.98 * pow(1.0 - max(-rd.y, 0.0), 5.0);
    if (F > 0.04) oCol = ssr(p, R);                     /* too faint to see otherwise */
}
);

/* Light shafts. The mask: sky seen near the sun, dimmed by cloud in front
 * of it (depth 0 is the far plane: nothing drawn there). */
static const char *FS_SHAFT_MASK = GLSL(
in vec2 vUV;
uniform vec4 uViewTan; uniform mat3 uCamRot;
uniform sampler2D uDepth; uniform sampler2D uCloudTex;
out vec4 oCol;
void main() {
    vec3 rd = normalize(uCamRot * vec3(uViewTan.x + vUV.x * uViewTan.z, uViewTan.y + vUV.y * uViewTan.w, -1.0));
    vec4 d = textureGather(uDepth, vUV);
    float sky = dot(vec4(equal(d, vec4(0.0))), vec4(0.25));
    float T = uCloudsOn > 0.5 ? texture(uCloudTex, vUV).a : 1.0;
    float mu = max(dot(rd, uSunDir), 0.0);
    oCol = vec4(uSunCol * sky * T * (pow(mu, 40.0) * 0.5 + pow(mu, 400.0) * 1.5), 1.0);
}
);

/* ...smeared out from the sun, fading with distance along each ray */
static const char *FS_SHAFT_BLUR = GLSL(
in vec2 vUV;
uniform sampler2D uShaft; uniform vec2 uSunUV; uniform float uAspect;
out vec4 oCol;
void main() {
    const int N = 40;
    vec2 d = uSunUV - vUV;
    /* reach at most ~0.7 of the screen toward the sun */
    float len = length(d * vec2(uAspect, 1.0));
    d *= min(1.0, 0.7 / max(len, 1e-4)) / float(N);
    vec2 p = vUV + d * ign(gl_FragCoord.xy);
    vec3 sum = vec3(0.0);
    float w = 1.0;
    for (int i = 0; i < N; i++) {
        sum += texture(uShaft, p).rgb * w;
        w *= 0.955;
        p += d;
    }
    oCol = vec4(sum / float(N), 1.0);
}
);

/* ---- GL state ---- */
static GLuint p_ssr, p_reflsrc, p_shaft_mask, p_shaft_blur, p_sky, p_cloud, p_tree_solid, p_terr, p_water, p_tree, p_bake, p_down, p_up, p_post;
static GLuint vao_empty, vao_water, tex_shadow, fbo_bake, tex_noise2, tex_noise3;
#define FOG_FAR 14000.0f

/* offscreen targets */
static GLuint fbo_scene, tex_scene, tex_depth, fbo_final, tex_final;
static GLuint fbo_refl, tex_refl, tex_refl_d;
static GLuint fbo_ssr, tex_ssr;                /* the water's reflections, quarter res */              /* half res copy of the scene for the water to reflect */
static GLuint fbo_shaft[2], tex_shaft[2];     /* light shafts at quarter res: the mask, then its radial blur */
static int shaft_w, shaft_h;
static GLuint fbo_cloud[2], tex_cloud[2];     /* sky clouds at half res, this frame's and last's (ping-pong) */
static int cloud_w, cloud_h;
static int cloud_cur, cloud_hist;             /* cloud_hist: tex_cloud[!cloud_cur] is valid */
static int scene_w, scene_h;
#define NBLOOM 5
static GLuint fbo_bloom[NBLOOM], tex_bloom[NBLOOM];
static int bloom_w[NBLOOM], bloom_h[NBLOOM];

static GLuint mk_tex2d(GLenum ifmt, int w, int h, GLenum fmt, GLenum type, const void *data, GLenum filt, GLenum wrap) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, ifmt, w, h, 0, fmt, type, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filt);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filt == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : filt);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap);
    return t;
}
static GLuint mk_fbo(GLuint tex) {
    GLuint f;
    glGenFramebuffers(1, &f);
    glBindFramebuffer(GL_FRAMEBUFFER, f);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    return f;
}

static void resize_scene(int w, int h) {
    if (w == scene_w && h == scene_h) return;
    scene_w = w; scene_h = h;
    if (tex_scene) {
        glDeleteTextures(1, &tex_scene); glDeleteTextures(1, &tex_depth); glDeleteFramebuffers(1, &fbo_scene);
        glDeleteTextures(2, tex_shaft); glDeleteFramebuffers(2, fbo_shaft);
        glDeleteTextures(1, &tex_refl); glDeleteTextures(1, &tex_refl_d); glDeleteFramebuffers(1, &fbo_refl);
        glDeleteTextures(1, &tex_ssr); glDeleteFramebuffers(1, &fbo_ssr);
        glDeleteTextures(NBLOOM, tex_bloom); glDeleteFramebuffers(NBLOOM, fbo_bloom);
        glDeleteTextures(2, tex_cloud); glDeleteFramebuffers(2, fbo_cloud);
    }
    for (int i = 0; i < 2; i++) {
        cloud_w = (w + 1) / 2; cloud_h = (h + 1) / 2;
        tex_cloud[i] = mk_tex2d(GL_RGBA16F, cloud_w, cloud_h, GL_RGBA, GL_FLOAT, NULL, GL_LINEAR, GL_CLAMP_TO_EDGE);
        fbo_cloud[i] = mk_fbo(tex_cloud[i]);
    }
    cloud_hist = 0;
    tex_scene = mk_tex2d(GL_R11F_G11F_B10F, w, h, GL_RGB, GL_FLOAT, NULL, GL_LINEAR, GL_CLAMP_TO_EDGE);   /* HDR; alpha unused */
    /* depth as a texture: the light shafts read where the sky shows through */
    tex_depth = mk_tex2d(GL_DEPTH_COMPONENT32F, w, h, GL_DEPTH_COMPONENT, GL_FLOAT, NULL, GL_NEAREST, GL_CLAMP_TO_EDGE);
    fbo_scene = mk_fbo(tex_scene);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, tex_depth, 0);
    tex_refl = mk_tex2d(GL_R11F_G11F_B10F, (w + 1) / 2, (h + 1) / 2, GL_RGB, GL_FLOAT, NULL, GL_LINEAR, GL_CLAMP_TO_EDGE);
    tex_refl_d = mk_tex2d(GL_R32F, (w + 1) / 2, (h + 1) / 2, GL_RED, GL_FLOAT, NULL, GL_NEAREST, GL_CLAMP_TO_EDGE);
    fbo_refl = mk_fbo(tex_refl);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, tex_refl_d, 0);
    static const GLenum bufs[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    glDrawBuffers(2, bufs);
    tex_ssr = mk_tex2d(GL_RGBA16F, (w + 3) / 4, (h + 3) / 4, GL_RGBA, GL_FLOAT, NULL, GL_LINEAR, GL_CLAMP_TO_EDGE);
    fbo_ssr = mk_fbo(tex_ssr);
    shaft_w = (w + 3) / 4; shaft_h = (h + 3) / 4;
    for (int i = 0; i < 2; i++) {
        tex_shaft[i] = mk_tex2d(GL_R11F_G11F_B10F, shaft_w, shaft_h, GL_RGB, GL_FLOAT, NULL, GL_LINEAR, GL_CLAMP_TO_EDGE);
        fbo_shaft[i] = mk_fbo(tex_shaft[i]);
    }
    for (int i = 0; i < NBLOOM; i++) {
        bloom_w[i] = w >> (i + 1); bloom_h[i] = h >> (i + 1);
        if (bloom_w[i] < 1) bloom_w[i] = 1;
        if (bloom_h[i] < 1) bloom_h[i] = 1;
        tex_bloom[i] = mk_tex2d(GL_R11F_G11F_B10F, bloom_w[i], bloom_h[i], GL_RGB, GL_FLOAT, NULL, GL_LINEAR, GL_CLAMP_TO_EDGE);
        fbo_bloom[i] = mk_fbo(tex_bloom[i]);
    }
}

/* tileable value-noise fbm, lattice period p at the base octave */
static float tile_vnoise(float x, float y, int p, uint32_t salt) {
    float fx = floorf(x), fy = floorf(y);
    int ix = (int)fx, iy = (int)fy;
    float tx = smooth(x - fx), ty = smooth(y - fy);
    int x0 = ((ix % p) + p) % p, x1 = (x0 + 1) % p, y0 = ((iy % p) + p) % p, y1 = (y0 + 1) % p;
    float a = rand01(x0, y0, salt), b = rand01(x1, y0, salt), c = rand01(x0, y1, salt), d = rand01(x1, y1, salt);
    return (a + (b - a) * tx) * (1 - ty) + (c + (d - c) * tx) * ty;
}
static int cmp_float_idx(const void *a, const void *b) {
    float x = ((const float *)a)[0], y = ((const float *)b)[0];
    return x < y ? -1 : x > y;
}
/* remap values to a uniform distribution, so thresholds mean area fractions */
static void equalize(float *v, int n, int stride, unsigned char *out, int ostride) {
    float *pairs = malloc(sizeof(float) * 2 * n);
    for (int i = 0; i < n; i++) { pairs[2 * i] = v[i * stride]; memcpy(&pairs[2 * i + 1], &i, sizeof(int)); }
    qsort(pairs, n, sizeof(float) * 2, cmp_float_idx);
    for (int r = 0; r < n; r++) {
        int i;
        memcpy(&i, &pairs[2 * r + 1], sizeof(int));
        out[i * ostride] = (unsigned char)(r * 255.0f / (n - 1) + 0.5f);
    }
    free(pairs);
}

/* Surface maps, tileable, 256^2, RG = slope (d/dx, d/dy, signed), B, A:
 *  tex_wave:   water, a sum of sine waves (integer wave vectors, so it
 *              tiles), mostly running with the wind; B = height
 *  tex_detail: ground, ridged fbm (crisp crevices); A = height, B = a
 *              softer fbm for colour
 * Mipmapped: the slopes average away with distance instead of sparkling. */
static GLuint tex_wave, tex_detail;
static void pack_slopes(const float *h, int n, float *gx, float *gy) {
    float gm = 1e-6f;
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++) {
            int i = y * n + x;
            gx[i] = 0.5f * (h[y * n + ((x + 1) & (n - 1))] - h[y * n + ((x - 1) & (n - 1))]);
            gy[i] = 0.5f * (h[((y + 1) & (n - 1)) * n + x] - h[((y - 1) & (n - 1)) * n + x]);
            gm = fmaxf(gm, fmaxf(fabsf(gx[i]), fabsf(gy[i])));
        }
    for (int i = 0; i < n * n; i++) { gx[i] /= gm; gy[i] /= gm; }
}
static GLuint upload_rgba(const float *r, const float *g, const float *b, const float *a, int n) {
    unsigned char *px = malloc((size_t)n * n * 4);
    for (int i = 0; i < n * n; i++) {
        px[4 * i + 0] = (unsigned char)lroundf((r[i] * 0.5f + 0.5f) * 255);
        px[4 * i + 1] = (unsigned char)lroundf((g[i] * 0.5f + 0.5f) * 255);
        px[4 * i + 2] = (unsigned char)lroundf(fminf(fmaxf(b[i], 0), 1) * 255);
        px[4 * i + 3] = (unsigned char)lroundf(fminf(fmaxf(a[i], 0), 1) * 255);
    }
    GLuint t = mk_tex2d(GL_RGBA8, n, n, GL_RGBA, GL_UNSIGNED_BYTE, px, GL_LINEAR_MIPMAP_LINEAR, GL_REPEAT);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY, 4.0f);
    free(px);
    return t;
}
static void norm01(float *v, int n) {
    float lo = 1e9f, hi = -1e9f;
    for (int i = 0; i < n; i++) { lo = fminf(lo, v[i]); hi = fmaxf(hi, v[i]); }
    for (int i = 0; i < n; i++) v[i] = (v[i] - lo) / (hi - lo);
}
static void make_surface_maps(void) {
    enum { N = 256, NW = 56 };
    float *h = malloc(sizeof(float) * N * N), *h2 = malloc(sizeof(float) * N * N);
    float *gx = malloc(sizeof(float) * N * N), *gy = malloc(sizeof(float) * N * N);
    /* water: wave vectors k (cycles per tile) mostly along +x, amplitude
     * falling as |k|^-1.6 like a wind-sea spectrum */
    float kx[NW], ky[NW], am[NW], ph[NW];
    for (int w = 0; w < NW; w++) {
        float kk = 3 + 37 * powf(rand01(w, 0, 950), 1.5f);
        float ang = (rand01(w, 1, 950) - 0.5f) * (rand01(w, 2, 950) < 0.8f ? 2.0f : 5.5f);
        kx[w] = roundf(kk * cosf(ang)); ky[w] = roundf(kk * sinf(ang));
        if (kx[w] == 0 && ky[w] == 0) kx[w] = 3;
        am[w] = powf(sqrtf(kx[w] * kx[w] + ky[w] * ky[w]), -1.6f) * (0.5f + rand01(w, 3, 950));
        ph[w] = rand01(w, 4, 950) * 6.2832f;
    }
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++) {
            float v = 0;
            for (int w = 0; w < NW; w++) v += am[w] * sinf(6.2832f * (kx[w] * x + ky[w] * y) / N + ph[w]);
            h[y * N + x] = v;
        }
    pack_slopes(h, N, gx, gy);
    norm01(h, N * N);
    tex_wave = upload_rgba(gx, gy, h, h, N);
    /* ground: ridged fbm for the crevices, a soft fbm for colour */
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++) {
            float s = 0, s2 = 0, amp = 0.5f;
            for (int o = 0, p = 8; o < 5; o++, p *= 2, amp *= 0.5f) {
                float r = 1 - fabsf(tile_vnoise((float)x / N * p, (float)y / N * p, p, 960 + o) * 2 - 1);
                s += r * r * amp;
                if (o < 3) s2 += tile_vnoise((float)x / N * p * 0.5f, (float)y / N * p * 0.5f, p / 2, 970 + o) * amp;
            }
            h[y * N + x] = s; h2[y * N + x] = s2;
        }
    pack_slopes(h, N, gx, gy);
    norm01(h, N * N); norm01(h2, N * N);
    tex_detail = upload_rgba(gx, gy, h2, h, N);
    free(h); free(h2); free(gx); free(gy);
}

static void make_noise(void) {
    enum { N2 = 256 };
    float *v = malloc(sizeof(float) * N2 * N2);
    unsigned char *px = malloc(N2 * N2 * 4);
    const int period[4] = {4, 8, 16, 8}, oct[4] = {6, 5, 5, 3};
    for (int ch = 0; ch < 4; ch++) {
        for (int y = 0; y < N2; y++)
            for (int x = 0; x < N2; x++) {
                float s = 0, amp = 0.5f, norm = 0;
                int p = period[ch];
                for (int o = 0; o < oct[ch] && p <= N2; o++, p *= 2, amp *= 0.5f) {
                    s += tile_vnoise((float)x / N2 * p, (float)y / N2 * p, p, 500 + ch * 16 + o) * amp;
                    norm += amp;
                }
                v[y * N2 + x] = s / norm;
            }
        equalize(v, N2 * N2, 1, px + ch, 4);
    }
    tex_noise2 = mk_tex2d(GL_RGBA8, N2, N2, GL_RGBA, GL_UNSIGNED_BYTE, px, GL_LINEAR_MIPMAP_LINEAR, GL_REPEAT);
    glGenerateMipmap(GL_TEXTURE_2D);
    free(v); free(px);

    /* 3D billows: inverted tileable Worley at three frequencies */
    enum { N3 = 64 };
    float *w = malloc(sizeof(float) * N3 * N3 * N3);
    unsigned char *vox = malloc(N3 * N3 * N3);
    const int cells[3] = {4, 8, 16};
    const float wt[3] = {0.6f, 0.28f, 0.12f};
    for (int z = 0; z < N3; z++)
        for (int y = 0; y < N3; y++)
            for (int x = 0; x < N3; x++) {
                float s = 0;
                for (int k = 0; k < 3; k++) {
                    int c = cells[k];
                    float px3 = (x + 0.5f) / N3 * c, py = (y + 0.5f) / N3 * c, pz = (z + 0.5f) / N3 * c;
                    int ix = (int)px3, iy = (int)py, iz = (int)pz;
                    float best = 9;
                    for (int dz = -1; dz <= 1; dz++)
                        for (int dy = -1; dy <= 1; dy++)
                            for (int dx = -1; dx <= 1; dx++) {
                                int cx = ix + dx, cy = iy + dy, cz = iz + dz;
                                int wx = (cx + c) % c, wy = (cy + c) % c, wz = (cz + c) % c;
                                uint32_t hh = hash2(wx + wz * 97, wy, 700 + k);
                                float fx = cx + (hh & 1023) / 1024.0f, fy = cy + ((hh >> 10) & 1023) / 1024.0f,
                                      fz = cz + rand01(wx + wz * 97, wy, 710 + k);
                                float d = (fx - px3) * (fx - px3) + (fy - py) * (fy - py) + (fz - pz) * (fz - pz);
                                if (d < best) best = d;
                            }
                    s += (1 - fminf(sqrtf(best), 1)) * wt[k];
                }
                w[(z * N3 + y) * N3 + x] = s;
            }
    equalize(w, N3 * N3 * N3, 1, vox, 1);
    glGenTextures(1, &tex_noise3);
    glBindTexture(GL_TEXTURE_3D, tex_noise3);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, N3, N3, N3, 0, GL_RED, GL_UNSIGNED_BYTE, vox);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
    glGenerateMipmap(GL_TEXTURE_3D);
    free(w); free(vox);
}

/* ---- trees ---- */
/* Meshes are in unit space: y scaled by the tree's height, x/z by its width.
 * Vertex: position, normal, ao (negative = bark). */
typedef struct { float *v; int n, cap; GLushort *idx; int ni; } Mesh;   /* v: n * 7 (pos, normal, ao); idx after mesh_index() */
static Mesh tmesh[4];                  /* pine near, pine far, broadleaf near, broadleaf far */
static GLuint vao_tree[4], vbo_tree_inst;

static void mesh_vert(Mesh *m, const float p[3], const float n[3], float ao) {
    if (m->n == m->cap) { m->cap = m->cap ? m->cap * 2 : 256; m->v = realloc(m->v, sizeof(float) * 7 * m->cap); }
    float *o = m->v + 7 * m->n++;
    memcpy(o, p, 12); memcpy(o + 3, n, 12); o[6] = ao;
}
/* a triangle wound counter-clockwise when seen from `out` */
static void mesh_tri(Mesh *m, const float *a, const float *b, const float *c,
                     const float *na, const float *nb, const float *nc, float aa, float ab, float ac, const float out[3]) {
    float u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, v[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    float fn[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
    if (fn[0] * out[0] + fn[1] * out[1] + fn[2] * out[2] < 0) {
        const float *t = b; b = c; c = t; t = nb; nb = nc; nc = t;
        float s = ab; ab = ac; ac = s;
    }
    mesh_vert(m, a, na, aa); mesh_vert(m, b, nb, ab); mesh_vert(m, c, nc, ac);
}
static void norm3(float v[3]) { float l = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); if (l > 0) { v[0] /= l; v[1] /= l; v[2] /= l; } }

static void gen_trunk(Mesh *m, float r, float y1, int seg) {
    for (int s = 0; s < seg; s++) {
        float a0 = s * 2 * (float)M_PI / seg, a1 = (s + 1) * 2 * (float)M_PI / seg;
        float p0[3] = {cosf(a0) * r, 0, sinf(a0) * r}, p1[3] = {cosf(a1) * r, 0, sinf(a1) * r};
        float q0[3] = {p0[0] * 0.7f, y1, p0[2] * 0.7f}, q1[3] = {p1[0] * 0.7f, y1, p1[2] * 0.7f};
        float n0[3] = {cosf(a0), 0, sinf(a0)}, n1[3] = {cosf(a1), 0, sinf(a1)};
        float out[3] = {cosf((a0 + a1) / 2), 0, sinf((a0 + a1) / 2)};
        mesh_tri(m, p0, p1, q0, n0, n1, n0, -0.5f, -0.5f, -0.9f, out);
        mesh_tri(m, p1, q1, q0, n1, n1, n0, -0.5f, -0.9f, -0.9f, out);
    }
}

/* tiers of branches: each a star-shaped skirt whose tips droop */
static void gen_pine(Mesh *m, int layers, int seg, int detail) {
    if (detail) gen_trunk(m, 0.035f, 0.35f, 5);
    for (int L = 0; L < layers; L++) {
        float f = (float)L / (layers - 1);
        float b = 0.10f + 0.62f * powf(f, 0.9f), t = b + 0.36f - 0.1f * f, r = 0.44f - 0.28f * f;
        if (L == layers - 1) t = 1.0f;
        float apex[3] = {0, t, 0}, up[3] = {0, 1, 0};
        for (int s = 0; s < seg; s++) {
            float a0 = (s + 0.5f * L) * 2 * (float)M_PI / seg, a1 = (s + 1 + 0.5f * L) * 2 * (float)M_PI / seg;
            /* even corners are branch tips (long, drooping), odd ones notches */
            int s1 = (s + 1) % seg;
            float k0 = s & 1 ? 0.72f : 1.08f, k1 = s1 & 1 ? 0.72f : 1.08f;
            float r0 = r * k0 * (0.9f + 0.2f * rand01(s, L, 800)), r1 = r * k1 * (0.9f + 0.2f * rand01(s1, L, 800));
            float d0 = s & 1 ? 0.03f : -0.05f, d1 = s1 & 1 ? 0.03f : -0.05f;
            float p0[3] = {cosf(a0) * r0, b + d0 * (t - b), sinf(a0) * r0};
            float p1[3] = {cosf(a1) * r1, b + d1 * (t - b), sinf(a1) * r1};
            float n0[3] = {cosf(a0) * (t - b), r * 1.2f, sinf(a0) * (t - b)}, n1[3] = {cosf(a1) * (t - b), r * 1.2f, sinf(a1) * (t - b)};
            norm3(n0); norm3(n1);
            float na[3] = {(n0[0] + n1[0]) * 0.3f, 1, (n0[2] + n1[2]) * 0.3f};
            norm3(na);
            float ao = 0.45f + 0.45f * f;
            float out[3] = {n0[0] + n1[0], n0[1] + n1[1], n0[2] + n1[2]};
            mesh_tri(m, p0, p1, apex, n0, n1, na, ao, ao, 1.0f, out);
            if (detail) {
                float c[3] = {0, b + 0.07f * (t - b), 0}, dn[3] = {0, -1, 0}, o2[3] = {0, -1, 0};
                mesh_tri(m, p0, p1, c, dn, dn, dn, 0.2f, 0.2f, 0.15f, o2);
            }
        }
        (void)up;
    }
}

/* lumpy canopy blob: an icosahedron (subdivided once for the near LOD) */
static void gen_blob(Mesh *m, const float ctr[3], const float rad[3], int subdiv, uint32_t salt) {
    const float X = 0.525731f, Z = 0.850651f;
    const float ico[12][3] = {{-X, 0, Z}, {X, 0, Z}, {-X, 0, -Z}, {X, 0, -Z}, {0, Z, X}, {0, Z, -X},
                              {0, -Z, X}, {0, -Z, -X}, {Z, X, 0}, {-Z, X, 0}, {Z, -X, 0}, {-Z, -X, 0}};
    const int f[20][3] = {{0, 4, 1}, {0, 9, 4}, {9, 5, 4}, {4, 5, 8}, {4, 8, 1}, {8, 10, 1}, {8, 3, 10}, {5, 3, 8},
                          {5, 2, 3}, {2, 7, 3}, {7, 10, 3}, {7, 6, 10}, {7, 11, 6}, {11, 0, 6}, {0, 1, 6},
                          {6, 1, 10}, {9, 0, 11}, {9, 11, 2}, {9, 2, 5}, {7, 2, 11}};
    float tris[80][3][3];
    int nt = 0;
    for (int i = 0; i < 20; i++) {
        const float *a = ico[f[i][0]], *b = ico[f[i][1]], *c = ico[f[i][2]];
        if (!subdiv) { memcpy(tris[nt][0], a, 12); memcpy(tris[nt][1], b, 12); memcpy(tris[nt][2], c, 12); nt++; continue; }
        float ab[3], bc[3], ca[3];
        for (int k = 0; k < 3; k++) { ab[k] = a[k] + b[k]; bc[k] = b[k] + c[k]; ca[k] = c[k] + a[k]; }
        norm3(ab); norm3(bc); norm3(ca);
        const float *sub[4][3] = {{a, ab, ca}, {ab, b, bc}, {ca, bc, c}, {ab, bc, ca}};
        for (int s = 0; s < 4; s++, nt++)
            for (int v = 0; v < 3; v++) memcpy(tris[nt][v], sub[s][v], 12);
    }
    for (int i = 0; i < nt; i++) {
        float p[3][3], n[3][3], ao[3], out[3] = {0, 0, 0};
        for (int v = 0; v < 3; v++) {
            const float *d = tris[i][v];
            int32_t qx = (int32_t)lroundf(d[0] * 1000), qy = (int32_t)lroundf(d[1] * 1000), qz = (int32_t)lroundf(d[2] * 1000);
            float lump = 0.82f + 0.32f * rand01(qx * 31 + qz, qy, salt);
            for (int k = 0; k < 3; k++) { p[v][k] = ctr[k] + d[k] * rad[k] * lump; n[v][k] = d[k]; out[k] += d[k]; }
            ao[v] = 0.35f + 0.65f * (d[1] * 0.5f + 0.5f);
        }
        /* smooth sphere normals (shared vertices); the leaf clumps in the
         * shader give the surface its texture */
        mesh_tri(m, p[0], p[1], p[2], n[0], n[1], n[2], ao[0], ao[1], ao[2], out);
    }
}

static void gen_broadleaf(Mesh *m, int detail) {
    if (detail) gen_trunk(m, 0.045f, 0.5f, 5);
    /* a main mass with lumps around and on top, so the outline is cloud-like */
    static const float lump[5][6] = {
        {0, 0.58f, 0, 0.38f, 0.28f, 0.38f},
        {0.10f, 0.80f, -0.06f, 0.26f, 0.20f, 0.26f},
        {-0.15f, 0.72f, 0.12f, 0.25f, 0.20f, 0.25f},
        {0.22f, 0.58f, 0.14f, 0.22f, 0.18f, 0.22f},
        {-0.12f, 0.55f, -0.24f, 0.22f, 0.18f, 0.22f},
    };
    for (int i = 0; i < (detail ? 5 : 2); i++) gen_blob(m, lump[i], lump[i] + 3, detail && i == 0, 820 + i);
}

/* merge identical vertices and index them, so each is shaded once per tree */
static void mesh_index(Mesh *m) {
    m->idx = malloc(sizeof(GLushort) * m->n);
    m->ni = m->n;
    int nu = 0;
    for (int i = 0; i < m->n; i++) {
        const float *a = m->v + 7 * i;
        int j = 0;
        while (j < nu && memcmp(m->v + 7 * j, a, sizeof(float) * 7)) j++;
        if (j == nu) memmove(m->v + 7 * nu++, a, sizeof(float) * 7);
        m->idx[i] = (GLushort)j;
    }
    m->n = nu;
}

/* Tree chunks: 128 m squares, generated once from the fine heightmap and kept
 * in a ring; each frame the visible ones are gathered into one instance buffer. */
#define TCH 128.0
#define TRING 64
#define TREE_R 1500.0f
#define TREE_NEAR 420.0f             /* near -> far mesh, crossfaded over +-TREE_BAND */
#define TREE_BAND 60.0f
#define THIN0 650.0f                     /* distance thinning, see trees_update() */
#define THIN1 1050.0f
#define FADE0 (TREE_R * 0.72f)            /* trees fade out (dithered) from here to TREE_R */
typedef struct { int32_t cx, cz; int n, valid; float *inst; } TChunk;   /* inst: n * 8 */
static TChunk tchunk[TRING * TRING];
static float *tinst;
static int tinst_cap;

static void tchunk_gen(TChunk *c, int32_t cx, int32_t cz) {
    c->cx = cx; c->cz = cz; c->n = 0; c->valid = 1;
    enum { G = 24 };
    float sp = (float)TCH / G;
    float *buf = malloc(sizeof(float) * 8 * G * G);
    for (int j = 0; j < G; j++)
        for (int i = 0; i < G; i++) {
            int32_t gi = cx * G + i, gj = cz * G + j;
            double x = cx * TCH + (i + 0.1f + 0.8f * rand01(gi, gj, 61)) * sp;
            double z = cz * TCH + (j + 0.1f + 0.8f * rand01(gi, gj, 62)) * sp;
            float h, fo, hx, hz, dummy;
            hm0_sample(x, z, &h, &fo);
            if (h < 3 || fo < 0.05f) continue;
            hm0_sample(x + 6, z, &hx, &dummy);
            hm0_sample(x, z + 6, &hz, &dummy);
            float slope = sqrtf((hx - h) * (hx - h) + (hz - h) * (hz - h)) / 6;
            if (slope > 0.75f) continue;
            if (rand01(gi, gj, 60) > fo * 0.9f * (1 - smoothstep(0.45f, 0.75f, slope))) continue;
            float pine_odds = h > 450 ? 0.92f : h > 180 ? 0.55f : 0.2f;
            int pine = rand01(gi, gj, 64) < pine_odds;
            float ht = pine ? 14 + 14 * rand01(gi, gj, 63) : 10 + 9 * rand01(gi, gj, 63);
            ht *= 0.75f + 0.25f * smoothstep(0.1f, 0.6f, fo);
            float *p = buf + 8 * c->n++;
            p[0] = (float)(x - cx * TCH); p[1] = h - 1.2f; p[2] = (float)(z - cz * TCH); p[3] = ht;
            p[4] = ht * (0.8f + 0.35f * rand01(gi, gj, 65)); p[5] = rand01(gi, gj, 66);
            p[6] = rand01(gi, gj, 67) * 6.2832f; p[7] = pine ? 0 : 1;
        }
    free(c->inst);
    c->inst = realloc(buf, sizeof(float) * 8 * (c->n ? c->n : 1));
}

/* per mesh (pine near/far, broadleaf near/far), solid then dithered: the
 * dithered ones (LOD crossfade, distance fade) need the shader that can
 * discard, which forgoes early depth rejection, so only they get it */
static int tree_count[8];

static int chunk_dist(const void *a, const void *b) {
    const int *p = a, *q = b;
    return (p[0] * p[0] + p[1] * p[1]) - (q[0] * q[0] + q[1] * q[1]);
}

/* gather visible trees into 4 LOD/type ranges; returns total */
static void trees_update(void) {
    int32_t ccx = (int32_t)floor(fl.x / TCH), ccz = (int32_t)floor(fl.z / TCH);
    int R = (int)ceil(TREE_R / TCH) + 1, gens = 0;
    float fx = -sinf(fl.yaw), fz = -cosf(fl.yaw);
    static float *lists[8];
    static int caps[8];
    int cnt[8] = {0};
    /* chunks nearest first, so the trees draw roughly front to back */
    static int (*ord)[2], nord;
    if (!ord) {
        ord = malloc(sizeof *ord * (2 * R + 1) * (2 * R + 1));
        for (int dz = -R; dz <= R; dz++)
            for (int dx = -R; dx <= R; dx++) { ord[nord][0] = dx; ord[nord][1] = dz; nord++; }
        qsort(ord, nord, sizeof *ord, chunk_dist);
    }
    for (int oi = 0; oi < nord; oi++) {
        int dx = ord[oi][0], dz = ord[oi][1];
        int32_t cx = ccx + dx, cz = ccz + dz;
        float ox = (float)(cx * TCH - fl.x), oz = (float)(cz * TCH - fl.z);
        float mx = ox + (float)TCH / 2, mz = oz + (float)TCH / 2, md = sqrtf(mx * mx + mz * mz);
        if (md > TREE_R + 92) continue;
        if (mx * fx + mz * fz < -140 && md > 200) continue;
        TChunk *c = &tchunk[(cz & (TRING - 1)) * TRING + (cx & (TRING - 1))];
        if (!c->valid || c->cx != cx || c->cz != cz) {
            if (gens >= 24 && md > 300) continue;
            tchunk_gen(c, cx, cz);
            gens++;
        }
        for (int t = 0; t < c->n; t++) {
            const float *s = c->inst + 8 * t;
            float x = ox + s[0], z = oz + s[2], d2 = x * x + z * z;
            if (d2 > TREE_R * TREE_R) continue;
            /* thin out with distance: half the trees shrink away between
             * THIN0 and THIN1 (each at its own distance), the rest grow
             * to cover for them, so the forest keeps its look for half
             * the vertices */
            float grow = 1, u = s[6] / 6.2832f;
            if (d2 > THIN0 * THIN0) {
                float d = sqrtf(d2);
                if (u < 0.5f) {
                    float gone = THIN0 + (THIN1 - THIN0) * u * 2;
                    if (d > gone) continue;
                    grow = fminf((gone - d) / 80.0f, 1.0f);
                } else grow = 1 + 0.41f * smoothstep(THIN0, THIN1, d);
            }
            /* in the crossfade band a tree goes in both lists */
            float lo = TREE_NEAR - TREE_BAND, hi = TREE_NEAR + TREE_BAND;
            int use[2] = {d2 < hi * hi, d2 > lo * lo};
            int dith = (d2 > lo * lo && d2 < hi * hi) || d2 > FADE0 * FADE0;
            for (int far = 0; far < 2; far++) {
                if (!use[far]) continue;
                int k = ((int)s[7] * 2 + far) * 2 + dith;
                if (cnt[k] == caps[k]) { caps[k] = caps[k] ? caps[k] * 2 : 4096; lists[k] = realloc(lists[k], sizeof(float) * 8 * caps[k]); }
                float *o = lists[k] + 8 * cnt[k]++;
                memcpy(o, s, sizeof(float) * 8);
                o[0] = x; o[2] = z; o[3] *= grow; o[4] *= grow;
            }
        }
    }
    int total = 0;
    for (int k = 0; k < 8; k++) total += cnt[k];
    if (total > tinst_cap) { tinst_cap = total * 2; tinst = realloc(tinst, sizeof(float) * 8 * tinst_cap); }
    int off = 0;
    for (int k = 0; k < 8; k++) {
        memcpy(tinst + 8 * off, lists[k], sizeof(float) * 8 * cnt[k]);
        tree_count[k] = cnt[k];
        off += cnt[k];
    }
    glBindBuffer(GL_ARRAY_BUFFER, vbo_tree_inst);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 8 * (total ? total : 1), NULL, GL_STREAM_DRAW);
    if (total) glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * 8 * total, tinst);
}

/* ---- drawing ---- */
static int W = 800, H = 600;

/* The virtual camera: canvas point (cx, cy) is straight ahead, cam_f is the
 * focal length in canvas units, and this window shows canvas rect view_*.
 * Canvas y grows downward. Without --layout the canvas is just the window. */
static float cam_cx, cam_cy, cam_f, view_x, view_y, view_w, view_h, hud_h;
static float lay_x0, lay_y0, lay_w, lay_h;

static void setup_view(void) {
    if (span_me < 0) {
        view_x = view_y = 0; view_w = W; view_h = H;
        cam_cx = W / 2.0f; cam_cy = H / 2.0f; hud_h = H;
        cam_f = H / 2.0f / tanf(opt_fov / 2 / DEG);
        lay_x0 = lay_y0 = 0; lay_w = W; lay_h = H;
        return;
    }
    float x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f;
    for (int i = 0; i < span_n; i++) {
        x0 = fminf(x0, span[i][0]); x1 = fmaxf(x1, span[i][0] + span[i][2]);
        y0 = fminf(y0, span[i][1]); y1 = fmaxf(y1, span[i][1] + span[i][3]);
    }
    lay_x0 = x0; lay_y0 = y0; lay_w = x1 - x0; lay_h = y1 - y0;
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

/* camera -> world rotation, column-major 3x3 */
static void cam_rot(float m[9]) {
    float cy = cosf(fl.yaw), sy = sinf(fl.yaw), cp = cosf(fl.pitch), sp = sinf(fl.pitch), cr = cosf(fl.roll), sr = sinf(fl.roll);
    /* Ry(yaw) * Rx(pitch) * Rz(roll) */
    float Ry[9] = {cy, 0, -sy, 0, 1, 0, sy, 0, cy};
    float Rx[9] = {1, 0, 0, 0, cp, sp, 0, -sp, cp};
    float Rz[9] = {cr, sr, 0, -sr, cr, 0, 0, 0, 1};
    float t[9];
    for (int c = 0; c < 3; c++)
        for (int r = 0; r < 3; r++) {
            t[c * 3 + r] = 0;
            for (int k = 0; k < 3; k++) t[c * 3 + r] += Rx[k * 3 + r] * Rz[c * 3 + k];
        }
    for (int c = 0; c < 3; c++)
        for (int r = 0; r < 3; r++) {
            m[c * 3 + r] = 0;
            for (int k = 0; k < 3; k++) m[c * 3 + r] += Ry[k * 3 + r] * t[c * 3 + k];
        }
}

static float view_tan[4], rot3[9], vp[16];

static void setup_matrices(void) {
    float l = (view_x - cam_cx) / cam_f, r = (view_x + view_w - cam_cx) / cam_f;
    float b = (cam_cy - view_y - view_h) / cam_f, t = (cam_cy - view_y) / cam_f;
    view_tan[0] = l; view_tan[1] = b; view_tan[2] = r - l; view_tan[3] = t - b;
    cam_rot(rot3);
    /* view = R^T * translate(0, -camY, 0); reversed-Z infinite projection */
    float V[16] = {0};
    for (int c = 0; c < 3; c++)
        for (int rr = 0; rr < 3; rr++) V[c * 4 + rr] = rot3[rr * 3 + c];
    for (int rr = 0; rr < 3; rr++) V[12 + rr] = -V[4 + rr] * fl.y;
    V[15] = 1;
    float n = 1.5f, P[16] = {0};
    P[0] = 2 / (r - l); P[8] = (r + l) / (r - l);
    P[5] = 2 / (t - b); P[9] = (t + b) / (t - b);
    P[14] = n; P[11] = -1;
    for (int c = 0; c < 4; c++)
        for (int rr = 0; rr < 4; rr++) {
            vp[c * 4 + rr] = 0;
            for (int k = 0; k < 4; k++) vp[c * 4 + rr] += P[k * 4 + rr] * V[c * 4 + k];
        }
}

static float sim_time;

static float pix_ang;                     /* radians per scene pixel */
static float mist_top0;                   /* the shaders' mistTop() under the camera */
/* mistTop(): the ground's average over ~2 km (the coarse map's mip 6,
 * filtered bilinearly like the GPU does) less 80 m */
static float mip6(int32_t kx, int32_t kz) {
    const HMap *m = &hm[1];
    double sum = 0;
    for (int j = 0; j < 64; j += 4)
        for (int i = 0; i < 64; i += 4)
            sum += m->data[2 * ((size_t)((kz * 64 + j) & (m->n - 1)) * m->n + ((kx * 64 + i) & (m->n - 1)))];
    return (float)(sum / 256);
}
static void mist_update(void) {
    double sp = hm[1].unit * BASE;
    /* the shader samples texel coordinate x / sp + 0.5; mip-6 texel centres sit at (k + 0.5) * 64 */
    double fx = (fl.x / sp + 0.5) / 64 - 0.5, fz = (fl.z / sp + 0.5) / 64 - 0.5;
    int32_t kx = (int32_t)floor(fx), kz = (int32_t)floor(fz);
    float tx = (float)(fx - kx), tz = (float)(fz - kz);
    float a = mip6(kx, kz), b = mip6(kx + 1, kz), c = mip6(kx, kz + 1), d = mip6(kx + 1, kz + 1);
    float h = (a + (b - a) * tx) * (1 - tz) + (c + (d - c) * tx) * tz;
    mist_top0 = h - 80;
}
static void set_common(GLuint p) {
    glUseProgram(p);
    glUniform3fv(UL(p, "uSunDir"), 1, pal.sun_dir);
    glUniform3fv(UL(p, "uSunCol"), 1, pal.sun);
    glUniform3fv(UL(p, "uZen"), 1, pal.zen);
    glUniform3fv(UL(p, "uHor"), 1, pal.hor);
    glUniform3fv(UL(p, "uHorSun"), 1, pal.hor_sun);
    glUniform3fv(UL(p, "uAmbSky"), 1, pal.amb_sky);
    glUniform3fv(UL(p, "uAmbGnd"), 1, pal.amb_gnd);
    glUniform1f(UL(p, "uCamY"), fl.y);
    glUniform1f(UL(p, "uTime"), fmodf(sim_time, 4096.0f));
    glUniform1f(UL(p, "uFogA"), pal.fog_a);
    glUniform1f(UL(p, "uMist"), pal.mist);
    glUniform1f(UL(p, "uMistTop0"), mist_top0);
    glUniform1f(UL(p, "uFogB"), pal.fog_b);
    glUniform1f(UL(p, "uHaze"), pal.haze);
    glUniform1f(UL(p, "uFogFar"), FOG_FAR);
    glUniform1f(UL(p, "uGlow"), pal.glow);
    glUniform1f(UL(p, "uStars"), pal.stars);
    glUniform1f(UL(p, "uDisc"), pal.disc);
    glUniform2f(UL(p, "uWorldMod"), (float)fmod(fl.x, 16384.0), (float)fmod(fl.z, 16384.0));
    /* wind: clouds drift ~6 m/s */
    double wx = fl.x + sim_time * 5.0, wz = fl.z + sim_time * 3.0;
    glUniform2f(UL(p, "uCloudMod"), (float)fmod(wx, 16384.0), (float)fmod(wz, 16384.0));
    glUniform1f(UL(p, "uCloudBase"), pal.cbase);
    glUniform1f(UL(p, "uCloudThick"), pal.cthick);
    glUniform1f(UL(p, "uCloudCover"), pal.cover);
    glUniform1f(UL(p, "uCloudDens"), pal.cdens);
    glUniform1f(UL(p, "uCloudsOn"), (float)opt_clouds);
    glUniform2f(UL(p, "uT0Cam"), (float)fmod(fl.x / BASE, 1024.0), (float)fmod(fl.z / BASE, 1024.0));
    glUniform2f(UL(p, "uT1Cam"), (float)fmod(fl.x / (4 * BASE), 1024.0), (float)fmod(fl.z / (4 * BASE), 1024.0));
    glUniform1f(UL(p, "uPixAng"), pix_ang);
    glUniformMatrix4fv(UL(p, "uVP"), 1, GL_FALSE, vp);
}

static int bake_row = -1;                 /* -1: bake everything next frame */
static void bake_step(void) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_bake);
    glViewport(0, 0, 1024, 1024);
    glUseProgram(p_bake);
    glUniform3fv(UL(p_bake, "uSunDir"), 1, pal.sun_dir);
    glUniform2i(UL(p_bake, "uOrg1"), hm[1].ox, hm[1].oz);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glBindVertexArray(vao_empty);
    if (bake_row >= 0) {
        glEnable(GL_SCISSOR_TEST);
        glScissor(0, bake_row, 1024, 64);
        bake_row = (bake_row + 64) & 1023;
    } else bake_row = 0;
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDisable(GL_SCISSOR_TEST);
}

#define PATCH_CELLS 64                 /* fine cells per terrain patch side: 512 m */
#define TESS_PX 6.0f                   /* target triangle width, pixels */
static GLuint ssbo_patch;
static int patch_np(void) { return 2 * (int)ceil(FOG_FAR / (BASE * PATCH_CELLS)) + 2; }
static int patch_dist(const void *a, const void *b) {
    const int *p = a, *q = b, c = patch_np() / 2;
    int da = (p[0] - c) * (p[0] - c) + (p[1] - c) * (p[1] - c), db = (q[0] - c) * (q[0] - c) + (q[1] - c) * (q[1] - c);
    return da - db;
}
static void patch_init(void) {
    int np = patch_np();
    int *o = malloc(sizeof(int) * 2 * np * np);
    for (int i = 0; i < np * np; i++) { o[2 * i] = i % np; o[2 * i + 1] = i / np; }
    qsort(o, np * np, sizeof(int) * 2, patch_dist);
    glGenBuffers(1, &ssbo_patch);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_patch);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(int) * 2 * np * np, o, GL_STATIC_DRAW);
    free(o);
}

static void draw_terrain(void) {
    set_common(p_terr);
    int32_t ccx = (int32_t)floor(fl.x / BASE), ccz = (int32_t)floor(fl.z / BASE);
    glUniform2i(UL(p_terr, "uCamCell"), ccx, ccz);
    glUniform2f(UL(p_terr, "uCamFrac"), (float)(fl.x - ccx * BASE), (float)(fl.z - ccz * BASE));
    /* the tessellation follows the shared virtual camera, not this instance's
     * render scale, so spanned monitors cut the ground identically */
    glUniform1f(UL(p_terr, "uTessK"), cam_f / TESS_PX);
    double ps = BASE * PATCH_CELLS;
    int np = patch_np();
    glUniform2i(UL(p_terr, "uPatch0"), (int32_t)floor(fl.x / ps) - np / 2, (int32_t)floor(fl.z / ps) - np / 2);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_patch);
    glUniform1i(UL(p_terr, "uPatchCells"), PATCH_CELLS);
    glPatchParameteri(GL_PATCH_VERTICES, 4);
    glBindVertexArray(vao_empty);
    glDrawArraysInstanced(GL_PATCHES, 0, 4, np * np);
}

static void draw_trees(void) {
    glEnable(GL_CULL_FACE);
    for (int dith = 0; dith < 2; dith++) {
        GLuint p = dith ? p_tree : p_tree_solid;
        set_common(p);
        glUniform1f(UL(p, "uTreeR"), TREE_R);
        glUniform1f(UL(p, "uTreeNear"), TREE_NEAR);
        glUniform1f(UL(p, "uTreeBand"), TREE_BAND);
        int off = 0;
        for (int r = 0; r < 8; r++) {
            int k = r >> 1;
            if (tree_count[r] && (r & 1) == dith) {
                glBindVertexArray(vao_tree[k]);
                glUniform1i(UL(p, "uFar"), k & 1);
                glDrawElementsInstancedBaseInstance(GL_TRIANGLES, tmesh[k].ni, GL_UNSIGNED_SHORT, NULL, tree_count[r], off);
            }
            off += tree_count[r];
        }
    }
    glDisable(GL_CULL_FACE);
}

/* any water in the coarse map ahead (a cheap scan): else skip the water
 * and its reflection passes altogether */
static int water_ahead(void) {
    const HMap *m = &hm[1];
    double sp = m->unit * BASE;
    float fx = -sinf(fl.yaw), fz = -cosf(fl.yaw);
    for (int j = -48; j <= 48; j++)
        for (int i = -48; i <= 48; i++) {
            float dx = i * 4 * (float)sp, dz = j * 4 * (float)sp;   /* every 4th texel: 128 m */
            if (dx * fx + dz * fz < -600) continue;                    /* well behind */
            int32_t tx = (int32_t)floor((fl.x + dx) / sp), tz = (int32_t)floor((fl.z + dz) / sp);
            if (m->data[2 * ((size_t)(tz & (m->n - 1)) * m->n + (tx & (m->n - 1)))] < 0.5f) return 1;
        }
    return 0;
}

static void draw_water(void) {
    set_common(p_water);
    glUniform2f(UL(p_water, "uSceneSize"), (float)scene_w, (float)scene_h);
    glUniform1f(UL(p_water, "uExtent"), FOG_FAR);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);
    glBindVertexArray(vao_water);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}

/* ---- birds ----
 * Flocks live in BCELL-metre world cells (most have one), each
 * circling a spot over the ground. Everything is a closed-form function of
 * the sim time, so spanned instances agree without simulating anything. */
#define BCELL 800.0
#define BIRD_R 2200.0f                       /* beyond this a bird is under a pixel */
static GLuint p_bird, vao_bird, vbo_bird_inst;
static int bird_nv, bird_n;

static void birds_init(void) {
    /* x right, y up, z forward (m, for a ~1.6 m span); w: body/inner/outer */
    static const float v[][4] = {
        /* body: nose, shoulders, tail fan */
        {0, 0, 0.35f, 0}, {-0.08f, 0, 0.05f, 0}, {0.08f, 0, 0.05f, 0},
        {-0.08f, 0, 0.05f, 0}, {0, 0.02f, -0.3f, 0}, {0.08f, 0, 0.05f, 0},
        {0, 0.02f, -0.3f, 0}, {-0.12f, 0, -0.42f, 0}, {0.12f, 0, -0.42f, 0},
        /* left wing: inner quad, outer triangle */
        {-0.08f, 0, 0.12f, 1}, {-0.45f, 0, 0.08f, 1}, {-0.08f, 0, -0.08f, 1},
        {-0.08f, 0, -0.08f, 1}, {-0.45f, 0, 0.08f, 1}, {-0.45f, 0, -0.14f, 1},
        {-0.45f, 0, 0.08f, 2}, {-0.82f, 0, -0.12f, 2}, {-0.45f, 0, -0.14f, 2},
        /* right wing */
        {0.08f, 0, 0.12f, 1}, {0.08f, 0, -0.08f, 1}, {0.45f, 0, 0.08f, 1},
        {0.08f, 0, -0.08f, 1}, {0.45f, 0, -0.14f, 1}, {0.45f, 0, 0.08f, 1},
        {0.45f, 0, 0.08f, 2}, {0.45f, 0, -0.14f, 2}, {0.82f, 0, -0.12f, 2},
    };
    bird_nv = (int)(sizeof v / sizeof v[0]);
    GLuint vbo;
    glGenVertexArrays(1, &vao_bird);
    glBindVertexArray(vao_bird);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof v, v, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, (void *)0);
    glGenBuffers(1, &vbo_bird_inst);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_bird_inst);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 32, (void *)0);
    glEnableVertexAttribArray(2); glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 32, (void *)16);
    glVertexAttribDivisor(1, 1);
    glVertexAttribDivisor(2, 1);
    glBindVertexArray(0);
}

static void birds_update(void) {
    enum { MAXB = 4096 };
    static float inst[MAXB * 8];
    double t = sim_time;
    int32_t ci = (int32_t)floor(fl.x / BCELL), cj = (int32_t)floor(fl.z / BCELL);
    bird_n = 0;
    for (int dj = -3; dj <= 3; dj++)
        for (int di = -3; di <= 3; di++) {
            int32_t gi = ci + di, gj = cj + dj;
            if (rand01(gi, gj, 900) > 0.85f) continue;
            /* the flock: circles radius R at speed v round a spot, often drifting off it */
            double cx = (gi + 0.2 + 0.6 * rand01(gi, gj, 901)) * BCELL, cz = (gj + 0.2 + 0.6 * rand01(gi, gj, 902)) * BCELL;
            float R = 80 + 220 * rand01(gi, gj, 903), v = 11 + 5 * rand01(gi, gj, 904);
            float dir = rand01(gi, gj, 905) < 0.5f ? 1.0f : -1.0f, w = dir * v / R;
            double ph = rand01(gi, gj, 906) * 6.2832 + w * t;
            double fx = cx + R * cos(ph) + 300 * sin(t * 0.013 + gi), fz = cz + R * sin(ph) + 300 * cos(t * 0.011 + gj);
            if ((fx - fl.x) * (fx - fl.x) + (fz - fl.z) * (fz - fl.z) > (BIRD_R + 150.0) * (BIRD_R + 150.0)) continue;
            /* velocity of the flock centre */
            double vx = -R * w * sin(ph) + 300 * 0.013 * cos(t * 0.013 + gi), vz = R * w * cos(ph) - 300 * 0.011 * sin(t * 0.011 + gj);
            /* ground here and a little ahead, from the fine map (it spans the bird range) */
            float g0, g1, fo;
            hm0_sample(fx, fz, &g0, &fo);
            hm0_sample(fx + vx * 6, fz + vz * 6, &g1, &fo);
            float gy = fmaxf(fmaxf(g0, g1), 0);
            float fy = gy + 60 + 90 * rand01(gi, gj, 907);
            int nb = 12 + (int)(48 * rand01(gi, gj, 908));
            float spread = 10 + 3.0f * sqrtf((float)nb);
            float roll = -dir * atanf(v * v / (R * 9.8f));
            for (int b = 0; b < nb && bird_n < MAXB; b++) {
                /* each bird wanders round its place in the flock */
                float p1 = rand01(b, gi * 7 + gj, 910) * 6.28f, p2 = rand01(b, gi * 7 + gj, 911) * 6.28f, p3 = rand01(b, gi * 7 + gj, 912) * 6.28f;
                float f1 = 0.15f + 0.1f * rand01(b, gj, 913), f2 = 0.11f + 0.1f * rand01(b, gi, 914);
                float ox = spread * sinf((float)(t * f1) + p1), oz = spread * sinf((float)(t * f2) + p2);
                float oy = spread * 0.3f * sinf((float)(t * (f1 + f2) * 0.5) + p3);
                double bvx = vx + spread * f1 * cos(t * f1 + p1), bvz = vz + spread * f2 * cos(t * f2 + p2);
                float bvy = spread * 0.3f * (f1 + f2) * 0.5f * cosf((float)(t * (f1 + f2) * 0.5) + p3);
                double bx = fx + ox, bz = fz + oz;
                float *o = inst + 8 * bird_n++;
                o[0] = (float)(bx - fl.x); o[1] = fy + oy; o[2] = (float)(bz - fl.z);
                o[3] = (float)atan2(-bvx, -bvz);
                /* flap in bursts, glide in between (wings slightly up) */
                float ft = (float)t + p3 * 3;
                float glide = smoothstep(0.2f, 0.6f, sinf(ft * 0.45f + p1));
                o[4] = (1 - glide) * 0.75f * sinf(ft * 17.0f) + glide * 0.12f;
                o[5] = roll + 0.15f * sinf(ft * 0.7f);
                o[6] = 1.4f + 0.5f * rand01(b, gi + gj, 915);             /* big birds: they have to read on screen */
                o[7] = atanf(bvy / (float)sqrt(bvx * bvx + bvz * bvz));
            }
        }
    glBindBuffer(GL_ARRAY_BUFFER, vbo_bird_inst);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 8 * (bird_n ? bird_n : 1), NULL, GL_STREAM_DRAW);
    if (bird_n) glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * 8 * bird_n, inst);
}

static void draw_birds(void) {
    if (!bird_n) return;
    set_common(p_bird);
    glBindVertexArray(vao_bird);
    glDrawArraysInstanced(GL_TRIANGLES, 0, bird_nv, bird_n);
}

static void fullscreen(void) {
    glBindVertexArray(vao_empty);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

/* returns the strength to composite at (0: sun off screen / behind) */
static float draw_shafts(void) {
    /* the sun on screen: into view space (camera looks down -z), then view uv */
    float v[3];
    for (int r = 0; r < 3; r++) v[r] = rot3[r * 3 + 0] * pal.sun_dir[0] + rot3[r * 3 + 1] * pal.sun_dir[1] + rot3[r * 3 + 2] * pal.sun_dir[2];
    if (v[2] > -0.05f) return 0;
    float u = (v[0] / -v[2] - view_tan[0]) / view_tan[2], w = (v[1] / -v[2] - view_tan[1]) / view_tan[3];
    float out = fmaxf(fmaxf(-u, u - 1), fmaxf(-w, w - 1));
    float amt = pal.shafts * (1 - smoothstep(0.1f, 0.8f, out)) * smoothstep(-0.05f, -0.25f, v[2]);
    if (amt <= 0) return 0;
    glViewport(0, 0, shaft_w, shaft_h);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_shaft[0]);
    set_common(p_shaft_mask);
    glUniform4fv(UL(p_shaft_mask, "uViewTan"), 1, view_tan);
    glUniformMatrix3fv(UL(p_shaft_mask, "uCamRot"), 1, GL_FALSE, rot3);
    glActiveTexture(GL_TEXTURE9);
    glBindTexture(GL_TEXTURE_2D, tex_depth);
    glActiveTexture(GL_TEXTURE8);
    glBindTexture(GL_TEXTURE_2D, tex_cloud[cloud_cur]);
    fullscreen();
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_shaft[1]);
    set_common(p_shaft_blur);
    glUniform2f(UL(p_shaft_blur, "uSunUV"), u, w);
    glUniform1f(UL(p_shaft_blur, "uAspect"), (float)shaft_w / shaft_h);
    glActiveTexture(GL_TEXTURE10);
    glBindTexture(GL_TEXTURE_2D, tex_shaft[0]);
    fullscreen();
    glBindTexture(GL_TEXTURE_2D, tex_shaft[1]);        /* for the composite */
    glActiveTexture(GL_TEXTURE7);
    return amt;
}

static void post(GLuint target) {
    glDisable(GL_DEPTH_TEST);
    float shaft_amt = draw_shafts();
    /* bloom: soft-threshold downsample chain, then tent upsample back */
    glUseProgram(p_down);
    glActiveTexture(GL_TEXTURE5);
    for (int i = 0; i < NBLOOM; i++) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_bloom[i]);
        glViewport(0, 0, bloom_w[i], bloom_h[i]);
        int sw = i ? bloom_w[i - 1] : scene_w, sh = i ? bloom_h[i - 1] : scene_h;
        glBindTexture(GL_TEXTURE_2D, i ? tex_bloom[i - 1] : tex_scene);
        glUniform2f(UL(p_down, "uTexel"), 1.0f / sw, 1.0f / sh);
        glUniform1f(UL(p_down, "uPrefilter"), i ? 0.0f : 1.2f);
        fullscreen();
    }
    glUseProgram(p_up);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    for (int i = NBLOOM - 1; i > 0; i--) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_bloom[i - 1]);
        glViewport(0, 0, bloom_w[i - 1], bloom_h[i - 1]);
        glBindTexture(GL_TEXTURE_2D, tex_bloom[i]);
        glUniform2f(UL(p_up, "uTexel"), 1.0f / bloom_w[i], 1.0f / bloom_h[i]);
        fullscreen();
    }
    glDisable(GL_BLEND);

    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glUseProgram(p_post);
    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D, tex_scene);
    glActiveTexture(GL_TEXTURE6);
    glBindTexture(GL_TEXTURE_2D, tex_bloom[0]);
    glActiveTexture(GL_TEXTURE7);       /* scratch unit for uploads */
    glUniform2f(UL(p_post, "uTexel"), 1.0f / scene_w, 1.0f / scene_h);
    glUniform1f(UL(p_post, "uExposure"), pal.exposure);
    glUniform1f(UL(p_post, "uBloomAmt"), 0.06f);
    glUniform1f(UL(p_post, "uShaftAmt"), shaft_amt);
    glUniform1f(UL(p_post, "uSat"), pal.sat);
    glUniform1f(UL(p_post, "uSharpen"), scene_w < W ? 0.5f : 0.0f);
    glUniform4f(UL(p_post, "uVig"), (view_x - lay_x0) / lay_w, (lay_y0 + lay_h - view_y - view_h) / lay_h, view_w / lay_w, view_h / lay_h);
    fullscreen();
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
    glUseProgram(0);
    glBindVertexArray(0);
    glMatrixMode(GL_PROJECTION); glLoadIdentity();
    glOrtho(ox, ox + view_w, oy, oy + view_h, -1, 1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    glDisable(GL_DEPTH_TEST);
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
    float top = hud_h / 2 - 60 * s, span_ = 150 * s, ppd = 7 * s;
    glBegin(GL_LINES);
    for (int d = (int)floorf(hdg - span_ / ppd); d <= (int)ceilf(hdg + span_ / ppd); d++) {
        if (d % 5) continue;
        float x = (d - hdg) * ppd;
        if (fabsf(x) > span_) continue;
        glVertex2f(x, top); glVertex2f(x, top + (d % 10 ? 7 : 13) * s);
    }
    glVertex2f(0, top - 4 * s); glVertex2f(-6 * s, top - 14 * s);
    glVertex2f(0, top - 4 * s); glVertex2f(6 * s, top - 14 * s);
    glEnd();
    for (int d = (int)floorf((hdg - span_ / ppd) / 10) * 10; d <= hdg + span_ / ppd; d += 10) {
        float x = (d - hdg) * ppd;
        if (fabsf(x) > span_ - 10 * s) continue;
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
}

static float render_scale = 1.0f;

static void draw_clouds(void) {
    static float prev_tan[4], prev_rot[9];
    static double prev_x, prev_y, prev_z;
    static unsigned frame;
    cloud_cur ^= 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_cloud[cloud_cur]);
    glViewport(0, 0, cloud_w, cloud_h);
    set_common(p_cloud);
    glUniform4fv(UL(p_cloud, "uViewTan"), 1, view_tan);
    glUniformMatrix3fv(UL(p_cloud, "uCamRot"), 1, GL_FALSE, rot3);
    glUniform4fv(UL(p_cloud, "uPrevTan"), 1, prev_tan);
    glUniformMatrix3fv(UL(p_cloud, "uPrevRot"), 1, GL_FALSE, prev_rot);
    glUniform3f(UL(p_cloud, "uPrevCam"), (float)(prev_x - fl.x), (float)(prev_y - fl.y), (float)(prev_z - fl.z));
    /* a big jump (seek, first frame) means the history is someone else's sky */
    double jump = fabs(prev_x - fl.x) + fabs(prev_z - fl.z);
    glUniform1f(UL(p_cloud, "uHistW"), cloud_hist && jump < 200 ? 0.85f : 0.0f);
    glUniform1f(UL(p_cloud, "uFrame"), (float)(frame++ & 1023));
    glUniform1f(UL(p_cloud, "uPixAng"), pix_ang * 2);        /* half res */
    glActiveTexture(GL_TEXTURE8);
    glBindTexture(GL_TEXTURE_2D, tex_cloud[!cloud_cur]);
    fullscreen();
    glBindTexture(GL_TEXTURE_2D, tex_cloud[cloud_cur]);     /* for the sky pass */
    glActiveTexture(GL_TEXTURE7);
    memcpy(prev_tan, view_tan, sizeof prev_tan);
    memcpy(prev_rot, rot3, sizeof prev_rot);
    prev_x = fl.x; prev_y = fl.y; prev_z = fl.z;
    cloud_hist = 1;
}

/* --bench: GPU time per pass, from timestamps between them */
enum { PR_BAKE, PR_CLOUD, PR_TREE, PR_TERR, PR_SKY, PR_WATER, PR_POST, PR_N };
static const char *prof_name[PR_N] = {"bake", "clouds", "trees", "terrain", "sky", "water", "post"};
static GLuint prof_q[PR_N + 1];
static double prof_ms[PR_N];
static int prof_on;
static void prof_mark(int i) { if (prof_on) glQueryCounter(prof_q[i], GL_TIMESTAMP); }
static void prof_collect(void) {
    if (!prof_on) return;
    GLuint64 t[PR_N + 1];
    for (int i = 0; i <= PR_N; i++) glGetQueryObjectui64v(prof_q[i], GL_QUERY_RESULT, &t[i]);
    for (int i = 0; i < PR_N; i++) prof_ms[i] += (t[i + 1] - t[i]) * 1e-6;
}

static void render(GLuint target) {
    setup_view();
    setup_matrices();
    hm_update(&hm[0], fl.x, fl.z);
    hm_update(&hm[1], fl.x, fl.z);
    if (opt_trees) trees_update();
    birds_update();
    mist_update();
    prof_mark(PR_BAKE);
    bake_step();

    int sw = (int)(W * render_scale + 0.5f), sh = (int)(H * render_scale + 0.5f);
    resize_scene(sw < 16 ? 16 : sw, sh < 16 ? 16 : sh);
    pix_ang = 1.0f / (cam_f * render_scale);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_scene);
    glViewport(0, 0, scene_w, scene_h);
    glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);
    glClearDepth(0);
    glClear(GL_DEPTH_BUFFER_BIT);
    glDepthFunc(GL_GREATER);

    prof_mark(PR_CLOUD);
    if (opt_clouds) draw_clouds();
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_scene);
    glViewport(0, 0, scene_w, scene_h);
    glEnable(GL_DEPTH_TEST);

    /* trees first: the terrain's costly shading then skips what they hide */
    prof_mark(PR_TREE);
    if (opt_trees) draw_trees();
    draw_birds();
    prof_mark(PR_TERR);
    draw_terrain();

    /* sky only where nothing was drawn: it sits at the far plane (depth 0) */
    prof_mark(PR_SKY);
    glDepthMask(GL_FALSE);
    glDepthFunc(GL_GEQUAL);
    set_common(p_sky);
    glUniform4fv(UL(p_sky, "uViewTan"), 1, view_tan);
    glUniformMatrix3fv(UL(p_sky, "uCamRot"), 1, GL_FALSE, rot3);
    fullscreen();
    glDepthFunc(GL_GREATER);
    glDepthMask(GL_TRUE);

    prof_mark(PR_WATER);
    if (water_ahead()) {
    /* what the water reflects: the scene so far, at half res */
    glDisable(GL_DEPTH_TEST);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_refl);
    glViewport(0, 0, (scene_w + 1) / 2, (scene_h + 1) / 2);
    glUseProgram(p_reflsrc);
    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D, tex_scene);
    glActiveTexture(GL_TEXTURE9);
    glBindTexture(GL_TEXTURE_2D, tex_depth);
    fullscreen();
    glActiveTexture(GL_TEXTURE11);
    glBindTexture(GL_TEXTURE_2D, tex_refl);
    glActiveTexture(GL_TEXTURE12);
    glBindTexture(GL_TEXTURE_2D, tex_refl_d);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_ssr);
    glViewport(0, 0, (scene_w + 3) / 4, (scene_h + 3) / 4);
    set_common(p_ssr);
    glUniform4fv(UL(p_ssr, "uViewTan"), 1, view_tan);
    glUniformMatrix3fv(UL(p_ssr, "uCamRot"), 1, GL_FALSE, rot3);
    fullscreen();
    glActiveTexture(GL_TEXTURE13);
    glBindTexture(GL_TEXTURE_2D, tex_ssr);
    glActiveTexture(GL_TEXTURE7);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_scene);
    glViewport(0, 0, scene_w, scene_h);
    glEnable(GL_DEPTH_TEST);
    draw_water();
    }

    glClipControl(GL_LOWER_LEFT, GL_NEGATIVE_ONE_TO_ONE);
    glDepthFunc(GL_LESS);
    prof_mark(PR_POST);
    post(target);
    if (opt_hud) draw_hud();
    prof_mark(PR_N);
}

static void gl_init(void) {
    const char *ver = (const char *)glGetString(GL_VERSION);
    int major = 0, minor = 0;
    if (!ver || sscanf(ver, "%d.%d", &major, &minor) != 2 || major * 10 + minor < 45) {
        fprintf(stderr, "flightsim: needs OpenGL 4.5 (got %s)\n", ver ? ver : "none");
        exit(1);
    }
    GLSL_COMMON = COMMON_SRC;
    p_sky = program("sky", VS_FULL, FS_SKY);
    p_cloud = program("cloud", VS_FULL, FS_CLOUD);
    p_shaft_mask = program("shaft mask", VS_FULL, FS_SHAFT_MASK);
    p_reflsrc = program("reflection source", VS_FULL, FS_REFLSRC);
    p_ssr = program("water reflections", VS_FULL, FS_SSR);
    p_shaft_blur = program("shaft blur", VS_FULL, FS_SHAFT_BLUR);
    p_terr = program_tess("terrain", VS_TERRAIN, TCS_TERRAIN, TES_TERRAIN, FS_TERRAIN);
    patch_init();
    p_water = program("water", VS_WATER, FS_WATER);
    p_tree = program_full("tree", VS_TREE, NULL, NULL, FS_TREE, "const bool kDither = true;\n");
    p_bird = program("bird", VS_BIRD, FS_BIRD);
    birds_init();
    p_tree_solid = program_full("tree", VS_TREE, NULL, NULL, FS_TREE, "layout(early_fragment_tests) in;\nconst bool kDither = false;\n");
    p_bake = program("bake", VS_FULL, FS_BAKE);
    p_down = program("down", VS_FULL, FS_DOWN);
    p_up = program("up", VS_FULL, FS_UP);
    p_post = program("post", VS_FULL, FS_POST);
    glGenVertexArrays(1, &vao_empty);

    /* heightmaps */
    for (int i = 0; i < 2; i++) {
        HMap *m = &hm[i];
        m->data = calloc((size_t)m->n * m->n * 2, sizeof(float));
        m->tex = mk_tex2d(GL_RG32F, m->n, m->n, GL_RG, GL_FLOAT, NULL, GL_LINEAR_MIPMAP_LINEAR, GL_REPEAT);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    tex_shadow = mk_tex2d(GL_RG8, 1024, 1024, GL_RG, GL_UNSIGNED_BYTE, NULL, GL_LINEAR, GL_REPEAT);
    fbo_bake = mk_fbo(tex_shadow);
    make_noise();
    make_surface_maps();
    GLuint units[5] = {hm[0].tex, hm[1].tex, tex_shadow, tex_noise2, tex_noise3};
    for (int i = 0; i < 5; i++) {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(i == 4 ? GL_TEXTURE_3D : GL_TEXTURE_2D, units[i]);
    }
    glActiveTexture(GL_TEXTURE14);
    glBindTexture(GL_TEXTURE_2D, tex_wave);
    glActiveTexture(GL_TEXTURE15);
    glBindTexture(GL_TEXTURE_2D, tex_detail);
    glActiveTexture(GL_TEXTURE7);       /* scratch unit for uploads */

    GLuint vbo;

    static const float wq[12] = {-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1};
    glGenVertexArrays(1, &vao_water);
    glBindVertexArray(vao_water);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof wq, wq, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, (void *)0);

    /* trees */
    gen_pine(&tmesh[0], 5, 10, 1);
    gen_pine(&tmesh[1], 3, 6, 0);
    gen_broadleaf(&tmesh[2], 1);
    gen_broadleaf(&tmesh[3], 0);
    glGenBuffers(1, &vbo_tree_inst);
    for (int k = 0; k < 4; k++) {
        mesh_index(&tmesh[k]);
        glGenVertexArrays(1, &vao_tree[k]);
        glBindVertexArray(vao_tree[k]);
        GLuint ibo;
        glGenBuffers(1, &ibo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLushort) * tmesh[k].ni, tmesh[k].idx, GL_STATIC_DRAW);
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 7 * tmesh[k].n, tmesh[k].v, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 28, (void *)0);
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 28, (void *)12);
        glEnableVertexAttribArray(2); glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 28, (void *)24);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_tree_inst);
        glEnableVertexAttribArray(3); glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 32, (void *)0);
        glEnableVertexAttribArray(4); glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 32, (void *)16);
        glVertexAttribDivisor(3, 1);
        glVertexAttribDivisor(4, 1);
    }
    glBindVertexArray(0);
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
        "                 [--time day|golden|dusk|night|random]\n"
        "                 [--speed X] [--fov DEG] [--seed N] [--epoch UNIXTIME]\n"
        "                 [--scale X]   (fixed render scale; default adapts to the frame budget)\n"
        "                 [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "                 [--shot FILE.ppm | --bench FRAMES] [--size WxH] [--at SECONDS]\n");
    exit(1);
}

/* fixed-step sim: the state at time t depends only on seed + t */
static const double STEP = 1.0 / 120;
static long steps;
static void advance(double t) {
    while ((steps + 1) * STEP <= t) {
        fly((float)STEP, (float)(steps * STEP));
        steps++;
    }
    sim_time = (float)(steps * STEP);
}

static void save_ppm(const char *path) {
    unsigned char *px = malloc((size_t)W * H * 3);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_final);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE, px);
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    for (int y = H - 1; y >= 0; y--) fwrite(px + (size_t)y * W * 3, 1, (size_t)W * 3, f);
    fclose(f);
    free(px);
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
        else if (!strcmp(a, "-scale") && i + 1 < argc) opt_scale = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-epoch") && i + 1 < argc) opt_epoch = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-layout") && i + 1 < argc) parse_layout(argv[++i]);
        else if (!strcmp(a, "-output") && i + 1 < argc) span_me = atoi(argv[++i]);
        else if (!strcmp(a, "-seed") && i + 1 < argc) opt_seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(a, "-shot") && i + 1 < argc) opt_shot = argv[++i];
        else if (!strcmp(a, "-bench") && i + 1 < argc) opt_bench = atoi(argv[++i]);
        else if (!strcmp(a, "-size") && i + 1 < argc) { if (sscanf(argv[++i], "%dx%d", &off_w, &off_h) != 2) usage(); }
        else if (!strcmp(a, "-at") && i + 1 < argc) opt_at = strtod(argv[++i], NULL);
        else usage();
    }
    if (span_me >= span_n) span_me = -1;
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    rng_state = opt_seed;
    seed_x = (frand() - 0.5f) * 2e5f;
    seed_z = (frand() - 0.5f) * 2e5f;
    int offscreen = opt_shot || opt_bench;
    long nc = sysconf(_SC_NPROCESSORS_ONLN);
    nthreads = nc > 1 ? (int)nc : 1;
    if (!offscreen && nthreads > 8) nthreads = 8;   /* be polite when there are several of us */

    Display *dpy = XOpenDisplay(NULL);
    if (!dpy) { fprintf(stderr, "flightsim: can't open display\n"); return 1; }
    int scr = DefaultScreen(dpy);
    Window root = RootWindow(dpy, scr);
    int attrs[] = {GLX_RGBA, GLX_DOUBLEBUFFER, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, None};
    XVisualInfo *vi = glXChooseVisual(dpy, scr, attrs);
    if (!vi) { fprintf(stderr, "flightsim: no suitable GLX visual\n"); return 1; }

    XWindowAttributes ra;
    XGetWindowAttributes(dpy, root, &ra);
    XSetWindowAttributes swa = {0};
    swa.colormap = XCreateColormap(dpy, root, vi->visual, AllocNone);
    swa.event_mask = StructureNotifyMask | KeyPressMask | ExposureMask;
    swa.override_redirect = opt_root;
    if (opt_root) { W = ra.width; H = ra.height; XSelectInput(dpy, root, StructureNotifyMask); }
    if (offscreen) { W = off_w; H = off_h; }
    Window win = XCreateWindow(dpy, root, 0, 0, offscreen ? 64 : W, offscreen ? 64 : H, 0, vi->depth, InputOutput, vi->visual,
                               CWColormap | CWEventMask | CWOverrideRedirect, &swa);
    XStoreName(dpy, win, "Flight Sim");
    XClassHint ch = {"flightsim", "Flightsim"};
    XSetClassHint(dpy, win, &ch);
    Atom wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(dpy, win, &wm_delete, 1);
    if (!offscreen) XMapRaised(dpy, win);

    GLXContext ctx = glXCreateContext(dpy, vi, NULL, True);
    glXMakeCurrent(dpy, win, ctx);
    choose_palette();
    gl_init();

    fl.speed = 150 * opt_speed;
    fl.yaw = frand() * 2 * (float)M_PI;
    fl.y = ground(0, 0) + 150;
    pick_route(0);

    if (offscreen) {
        tex_final = mk_tex2d(GL_RGBA8, W, H, GL_RGBA, GL_UNSIGNED_BYTE, NULL, GL_NEAREST, GL_CLAMP_TO_EDGE);
        fbo_final = mk_fbo(tex_final);
        render_scale = opt_scale > 0 ? opt_scale : 1.0f;
        advance(opt_at);
        render(fbo_final);                 /* warm-up: streams everything, full bake */
        glFinish();
        if (opt_shot) {
            for (int i = 0; i < 24; i++) render(fbo_final);   /* let the cloud accumulation settle */
            save_ppm(opt_shot);
            return 0;
        }
        double t0 = now(), worst = 0;
        glGenQueries(PR_N + 1, prof_q);
        prof_on = 1;
        for (int i = 0; i < opt_bench; i++) {
            double f0 = now();
            advance(opt_at + (i + 1) / 60.0);
            render(fbo_final);
            glFinish();
            prof_collect();
            worst = fmax(worst, now() - f0);
        }
        double per = (now() - t0) / opt_bench;
        printf("%dx%d scale %.2f: %.2f ms/frame avg, %.2f worst (%s)\n", W, H, render_scale, per * 1e3, worst * 1e3,
               (const char *)glGetString(GL_RENDERER));
        for (int i = 0; i < PR_N; i++) printf("  %-8s %6.2f ms\n", prof_name[i], prof_ms[i] / opt_bench);
        return 0;
    }

    /* Fixed-step sim clocked off the wall: the state at time t depends only on
     * seed + epoch + t, so separate instances stay in lockstep. */
    double epoch = opt_epoch ? opt_epoch : wallclock();
    render_scale = opt_scale > 0 ? opt_scale : 0.75f;
    double avg = 1.0 / 60, last_change = 0;
    for (int frame = 0;; frame++) {
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
        advance(wallclock() - epoch + opt_at);
        render(0);
        glFinish();
        double took = now() - t;
        glXSwapBuffers(dpy, win);

        /* dynamic resolution: keep each frame well inside a 60 Hz budget,
         * leaving room for the other instances sharing the GPU */
        if (frame > 10) avg += (took - avg) * 0.1;
        if (opt_scale <= 0 && frame > 30 && t - last_change > 0.5) {
            float s = render_scale;
            if (avg > 0.0125) s *= 0.9f;
            else if (avg < 0.008) s *= 1.06f;
            s = s < 0.35f ? 0.35f : s > 1.0f ? 1.0f : s;
            if (fabsf(s - render_scale) > 0.02f) { render_scale = s; last_change = t; avg = 1.0 / 90; }
        }
        double spare = 1.0 / 60 - (now() - t);
        if (spare > 0) {
            struct timespec ts = {0, (long)(spare * 1e9)};
            nanosleep(&ts, NULL);
        }
    }
}
