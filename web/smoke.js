// web/smoke.js
//
// Headless Chromium smoke test for the Singularity WebGPU demo.
// Runs via puppeteer with SwiftShader Vulkan so it works on GitHub's
// GPU-less ubuntu-latest runners. Writes the captured canvas to a PNG
// and asserts the image isn't all-black (rendering actually happened).
//
// Invoked from CI after python3 -m http.server is serving build-wasm/web.
// Exit codes: 0 = pass, non-zero = fail (prints all console logs for
// debugging).

const fs = require('fs');
const puppeteer = require('puppeteer');

const URL = process.env.SINGULARITY_WEB_URL || 'http://127.0.0.1:8123/';
const OUT = process.env.SINGULARITY_WEB_OUT || '/tmp/singularity-smoke.png';
const WAIT_MS = Number(process.env.SINGULARITY_WEB_WAIT_MS || 6000);

(async () => {
  const browser = await puppeteer.launch({
    headless: 'new',
    args: [
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan',
      '--use-vulkan=swiftshader',
      '--no-sandbox',
    ],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1024, height: 600 });

  const logs = [];
  page.on('console', m => logs.push('[' + m.type() + '] ' + m.text()));
  page.on('pageerror', e => logs.push('[pageerror] ' + e.message));

  await page.goto(URL, { waitUntil: 'networkidle0' });
  await new Promise(r => setTimeout(r, WAIT_MS));

  // The device-ready banner prints to stderr; also check it lands.
  const deviceReady = logs.some(l => l.includes('device ready'));

  const dataUrl = await page.evaluate(() => {
    const c = document.getElementById('canvas');
    if (!c) return null;
    try {
      return c.toDataURL('image/png');
    } catch (e) {
      return 'error:' + e.message;
    }
  });

  let ok = false;
  let pngBytes = 0;
  if (dataUrl && dataUrl.startsWith('data:image/png;base64,')) {
    const b64 = dataUrl.slice('data:image/png;base64,'.length);
    const buf = Buffer.from(b64, 'base64');
    pngBytes = buf.length;
    fs.writeFileSync(OUT, buf);
    // PNG size as a proxy for "something was rendered". A blank black
    // canvas compresses to ~500 bytes; the Kerr + lensed disc render
    // at 960x540 lands in the 150-250 KB range. 20 KB threshold catches
    // even partial renders (geodesic pass only, no bloom, etc.) without
    // false-positiving on a blank frame.
    ok = pngBytes > 20_000;
  }

  console.log('device_ready:', deviceReady);
  console.log('png_bytes:', pngBytes);
  console.log('png_written:', OUT);
  if (!ok) {
    console.log('---console log---');
    console.log(logs.slice(0, 80).join('\n'));
  }

  await browser.close();
  process.exit(ok && deviceReady ? 0 : 1);
})().catch(err => {
  console.error('smoke.js crashed:', err);
  process.exit(2);
});
