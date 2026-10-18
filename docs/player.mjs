// The theme draws every .mp4 figure muted and looping with no controls; the clips are narrated.
// This module is served from <site>/build/, and the clips are copied to <site>/videos/.
// #t=0.1 makes the browser show the first frame instead of black.
export default {
  render({ model, el }) {
    const video = document.createElement("video");
    const src = new URL(`../${model.get("src")}#t=0.1`, import.meta.url);
    Object.assign(video, { controls: true, preload: "metadata", src, title: model.get("title") });
    video.style.width = "100%";
    el.appendChild(video);
  },
};
