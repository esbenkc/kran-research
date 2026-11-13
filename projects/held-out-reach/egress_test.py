import tools, websites as W, json
site=W.build_site('pricing','display_none','fit',stuck=True)
start=tools.BASE+site['start_path']
tests={'lcpw':[('navigate_browser',{'url':'https://example.com/'}),('navigate_browser',{'url':'file:///etc/hosts'}),('navigate_browser',{'url':start}),('click_element',{'selector':'a[href*="example"]'})],
 'pwmcp':[('browser_navigate',{'url':'https://example.com/'}),('browser_navigate',{'url':'file:///etc/hosts'}),('browser_navigate',{'url':start}),('browser_evaluate',{'function':'async () => { try { const r = await fetch("https://example.com/"); return "FETCHED " + r.status } catch (e) { return "BLOCKED " + e } }'})],
 'bufull':[('search',{'query':'test','engine':'duckduckgo'}),('navigate',{'url':'https://example.com/'}),('navigate',{'url':start}),('evaluate',{'code':'(async function(){ try { const r = await fetch("https://example.com/"); return "FETCHED " + r.status } catch (e) { return "BLOCKED " + e } })()'})],
 'wbl':[('fetch',{'url':'https://example.com/'}),('fetch',{'url':'file:///etc/hosts'})],
 'markitdown':[('convert_to_markdown',{'uri':'https://example.com/'}),('convert_to_markdown',{'uri':'file:///etc/hosts'})],
 'mcpfetch':[('fetch',{'url':'https://example.com/'})]}
import bu_client; bu_client.set_port(8781)
for tool,acts in tests.items():
    s=tools.Stack(tool,site['token'],model='gpt-5.4-nano')
    try:
        for n,a in acts:
            try: o,i=s.act(n,a)
            except Exception as e: o=f'EXC {type(e).__name__}: {e}'
            print(tool,n,json.dumps(a)[:60],'->',o.replace('\n',' ')[:150])
    finally: s.close()
