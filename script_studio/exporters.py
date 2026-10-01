"""Export a ScriptResult as Markdown, Fountain screenplay, CSV shot list, plain text, JSON or PDF."""
from __future__ import annotations

import csv
import io

from .models import ScriptResult
from .planner import fmt_time


def _inc_label(r: ScriptResult) -> str:
    return 'Complete script' if r.settings.increment == 'full' else f'{r.settings.increment}-second segments'


def to_json(r: ScriptResult) -> str:
    return r.model_dump_json(indent=2)


def to_markdown(r: ScriptResult) -> str:
    b = r.bible
    out = [f'# {b.title}', '', f'*{b.logline}*', '',
           f'**Format:** {_inc_label(r)} · **Running time:** {fmt_time(r.plan.total_seconds)} · '
           f'**Genre:** {b.genre} · **Tone:** {b.tone} · **Aspect:** {r.settings.aspect_ratio}', '',
           f'**Visual style:** {b.visual_style}', '', '## Characters']
    out += [f'- **{c.name}** ({c.role}): {c.visual_signature}. Voice: {c.voice}' for c in b.characters]
    out += ['', '## Locations'] + [f'- **{l.name}**: {l.visual_signature}' for l in b.locations]
    out += ['', '## Script', '']
    for s in r.segments:
        p = r.plan.segments[s.index]
        out += [f'### {s.index + 1}. {fmt_time(p.start)} – {fmt_time(p.end)} · {s.scene_heading}', '', f'_{s.summary}_', '']
        for k, sh in enumerate(s.shots, 1):
            out += [f'**Shot {s.index + 1}.{k}** ({fmt_time(sh.start)}–{fmt_time(sh.end)}, {sh.camera})', '',
                    f'> {sh.prompt}', '', sh.action]
            for d in sh.dialogue:
                out.append(f'- **{d.character.upper()}**' + (f' ({d.delivery})' if d.delivery else '') + f': {d.line}')
            for label, val in (('V.O.', sh.voiceover), ('Lyrics', sh.lyrics), ('Sound', sh.sound), ('VFX', sh.vfx), ('Negative', sh.negative)):
                if val:
                    out.append(f'- *{label}:* {val}')
            out.append('')
        out += [f'*Transition:* {s.transition_out}', '', '---', '']
    return '\n'.join(out)


def to_fountain(r: ScriptResult) -> str:
    b = r.bible
    out = [f'Title: {b.title}', f'Credit: Adapted with Script Studio', f'Notes: {b.logline}', '', '']
    last_heading = None
    for s in r.segments:
        heading = s.scene_heading.strip().upper()
        if heading and heading != last_heading:
            if not heading.startswith(('INT', 'EXT', 'I/E', 'EST')):
                heading = '.' + heading
            out += [heading, '']
            last_heading = heading
        out += [f'[[{fmt_time(r.plan.segments[s.index].start)} – {s.summary}]]', '']
        for sh in s.shots:
            out += [f'[[{sh.camera.upper()} {fmt_time(sh.start)}-{fmt_time(sh.end)}: {sh.prompt}]]', '', sh.action, '']
            for d in sh.dialogue:
                out.append(d.character.upper())
                if d.delivery:
                    out.append(f'({d.delivery})')
                out += [d.line, '']
            if sh.voiceover:
                out += ['NARRATOR (V.O.)', sh.voiceover, '']
            if sh.lyrics:
                out += ['~' + sh.lyrics.replace('\n', '\n~'), '']
            if sh.sound:
                out += [f'[[SOUND: {sh.sound}]]', '']
        if s.transition_out:
            t = s.transition_out.strip()
            out += [('> ' + t.upper()) if len(t) < 40 else f'[[TRANSITION: {t}]]', '']
    return '\n'.join(out)


def to_csv(r: ScriptResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['segment', 'shot', 'start', 'end', 'duration', 'scene', 'camera', 'prompt', 'action', 'dialogue',
                'voiceover', 'lyrics', 'sound', 'vfx', 'negative'])
    for s in r.segments:
        for k, sh in enumerate(s.shots, 1):
            w.writerow([s.index + 1, k, f'{sh.start:.2f}', f'{sh.end:.2f}', f'{sh.end - sh.start:.2f}', s.scene_heading,
                        sh.camera, sh.prompt, sh.action,
                        ' / '.join(f'{d.character}: {d.line}' for d in sh.dialogue), sh.voiceover, sh.lyrics, sh.sound,
                        sh.vfx, sh.negative])
    return buf.getvalue()


def to_prompts(r: ScriptResult) -> str:
    """Just the generator prompts, one per shot, in order: handy for batch generation."""
    out = []
    for s in r.segments:
        for k, sh in enumerate(s.shots, 1):
            out.append(f'# {s.index + 1}.{k}  {fmt_time(sh.start)}-{fmt_time(sh.end)}  ({sh.end - sh.start:.1f}s)\n{sh.prompt}'
                       + (f'\nNegative: {sh.negative}' if sh.negative else ''))
    return '\n\n'.join(out) + '\n'


def to_pdf(r: ScriptResult) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from xml.sax.saxutils import escape as esc

    ink, gold, muted, card = colors.HexColor('#16181d'), colors.HexColor('#b8862a'), colors.HexColor('#5d6272'), colors.HexColor('#f6f3ec')
    S = {
        'title': ParagraphStyle('t', fontName='Helvetica-Bold', fontSize=22, leading=27, textColor=ink, spaceAfter=6),
        'sub': ParagraphStyle('s', fontName='Helvetica-Oblique', fontSize=10.5, leading=14, textColor=muted, spaceAfter=10),
        'h2': ParagraphStyle('h2', fontName='Helvetica-Bold', fontSize=13, leading=17, textColor=ink, spaceBefore=10, spaceAfter=4),
        'body': ParagraphStyle('b', fontName='Helvetica', fontSize=9.3, leading=13, textColor=ink, spaceAfter=4),
        'scene': ParagraphStyle('sc', fontName='Courier-Bold', fontSize=10.5, leading=14, textColor=ink, spaceBefore=8, spaceAfter=2),
        'time': ParagraphStyle('tm', fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=gold),
        'prompt': ParagraphStyle('p', fontName='Helvetica', fontSize=8.8, leading=12, textColor=colors.HexColor('#2a2d35')),
        'action': ParagraphStyle('a', fontName='Courier', fontSize=9.5, leading=12.5, textColor=ink, spaceBefore=3),
        'char': ParagraphStyle('c', fontName='Courier-Bold', fontSize=9.5, leading=12, leftIndent=1.6 * inch, spaceBefore=4),
        'paren': ParagraphStyle('pa', fontName='Courier', fontSize=9.5, leading=12, leftIndent=1.3 * inch),
        'line': ParagraphStyle('l', fontName='Courier', fontSize=9.5, leading=12, leftIndent=1.0 * inch, rightIndent=1.0 * inch),
        'meta': ParagraphStyle('m', fontName='Helvetica', fontSize=8, leading=10.5, textColor=muted),
    }
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=0.8 * inch, rightMargin=0.8 * inch, topMargin=0.7 * inch,
                            bottomMargin=0.7 * inch, title=r.bible.title, author='Script Studio')
    cw = letter[0] - 1.6 * inch
    b = r.bible
    story = [Paragraph(esc(b.title), S['title']), Paragraph(esc(b.logline), S['sub']),
             Paragraph(esc(f'{_inc_label(r)} · {fmt_time(r.plan.total_seconds)} · {b.genre} · {b.tone} · {r.settings.aspect_ratio}'), S['meta']),
             Spacer(1, 6), Paragraph('<b>Visual style:</b> ' + esc(b.visual_style), S['body']),
             Paragraph('Characters', S['h2'])]
    story += [Paragraph(f'<b>{esc(c.name)}</b> ({esc(c.role)}): {esc(c.visual_signature)}', S['body']) for c in b.characters]
    story += [Paragraph('Locations', S['h2'])]
    story += [Paragraph(f'<b>{esc(l.name)}</b>: {esc(l.visual_signature)}', S['body']) for l in b.locations]
    story += [Paragraph('Script', S['h2'])]
    for s in r.segments:
        p = r.plan.segments[s.index]
        story += [Paragraph(esc(f'{s.index + 1}.  {s.scene_heading.upper()}'), S['scene']),
                  Paragraph(esc(f'{fmt_time(p.start)} – {fmt_time(p.end)}  ·  {s.summary}'), S['meta'])]
        for k, sh in enumerate(s.shots, 1):
            box = Table([[[Paragraph(esc(f'SHOT {s.index + 1}.{k}  ·  {fmt_time(sh.start)}–{fmt_time(sh.end)}  ·  {sh.camera.upper()}'), S['time']),
                           Paragraph(esc(sh.prompt), S['prompt'])]]], colWidths=[cw])
            box.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), card), ('LINEBEFORE', (0, 0), (0, -1), 2.5, gold),
                                     ('LEFTPADDING', (0, 0), (-1, -1), 8), ('TOPPADDING', (0, 0), (-1, -1), 4),
                                     ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
            block = [box, Paragraph(esc(sh.action), S['action'])]
            for d in sh.dialogue:
                block.append(Paragraph(esc(d.character.upper()), S['char']))
                if d.delivery:
                    block.append(Paragraph(esc(f'({d.delivery})'), S['paren']))
                block.append(Paragraph(esc(d.line), S['line']))
            if sh.voiceover:
                block += [Paragraph('NARRATOR (V.O.)', S['char']), Paragraph(esc(sh.voiceover), S['line'])]
            if sh.lyrics:
                block.append(Paragraph('<i>♪ ' + esc(sh.lyrics) + ' ♪</i>', S['line']))
            extra = ' · '.join(f'{k_}: {v}' for k_, v in (('Sound', sh.sound), ('VFX', sh.vfx), ('Negative', sh.negative)) if v)
            if extra:
                block.append(Paragraph(esc(extra), S['meta']))
            block.append(Spacer(1, 5))
            story.append(KeepTogether(block))
        if s.transition_out:
            story.append(Paragraph(esc(s.transition_out.upper()), ParagraphStyle('tr', parent=S['action'], alignment=2)))
    doc.build(story)
    return buf.getvalue()


FORMATS = {
    'md': ('text/markdown', '.md', to_markdown),
    'fountain': ('text/plain', '.fountain', to_fountain),
    'csv': ('text/csv', '.csv', to_csv),
    'prompts': ('text/plain', '-prompts.txt', to_prompts),
    'json': ('application/json', '.json', to_json),
    'pdf': ('application/pdf', '.pdf', to_pdf),
}
