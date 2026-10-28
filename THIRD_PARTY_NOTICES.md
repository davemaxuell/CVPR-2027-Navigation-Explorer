# Third-party notices

The Python viewer and interface are extracted from the NavVEM research workspace,
whose MIT license is retained in [LICENSE](LICENSE). This repository does not
redistribute datasets, captured images, building meshes, model checkpoints, or
private review notes. The bundled episodes describe synthetic scenes.

The browser dependencies are vendored so replay does not require a CDN:

| Component | Version | License and source |
| --- | --- | --- |
| Three.js and example controls/renderers | r170 | MIT; [included notice](src/navigation_explorer/static/vendor/Three-LICENSE.txt), [upstream](https://github.com/mrdoob/three.js/tree/r170) |
| Inter Variable | 4.1 | SIL Open Font License 1.1; [included notice](src/navigation_explorer/static/ui/Inter-LICENSE.txt), [upstream](https://github.com/rsms/inter/tree/v4.1) |
| Lucide SVG icons | 0.468.0 | ISC/MIT; [included notice](src/navigation_explorer/static/ui/Lucide-LICENSE.txt), [upstream](https://github.com/lucide-icons/lucide/tree/0.468.0) |

R2R, RxR, REVERIE, Matterport3D, ScaleVLN/DUET, and ETPNav names identify compatible
formats or sources; their datasets, code, pretrained weights, and terms remain
with their respective projects. The ETPNav adapter reads recorded evaluations and
does not import or bundle Habitat or ETPNav inference code.
