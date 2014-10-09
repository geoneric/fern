<map version="0.9.0">
<!-- To view this file, download free mind mapping software FreeMind from http://freemind.sourceforge.net -->
<node CREATED="1383638431576" ID="ID_1502853423" MODIFIED="1383638454050" TEXT="IO">
<node CREATED="1383638473261" ID="ID_1975117359" MODIFIED="1383638480553" POSITION="right" TEXT="Inputs">
<node CREATED="1383638481661" ID="ID_235728244" MODIFIED="1383638853272" TEXT="Undefined identifiers">
<node CREATED="1383638576548" ID="ID_1713933001" MODIFIED="1383638589468" TEXT="These must be resolved before execution"/>
<node CREATED="1383639872687" ID="ID_680799248" MODIFIED="1383639893513" TEXT="Multiple undefined identifiers with the same name should be handled as one read"/>
<node CREATED="1383639745795" ID="ID_127186836" MODIFIED="1383639751905" TEXT="Annotation">
<node CREATED="1383639758581" ID="ID_778758919" MODIFIED="1383639775309" TEXT="Pass symbol table with expression types so the tree can be annotated"/>
</node>
<node CREATED="1383639752372" ID="ID_1162424375" MODIFIED="1383639755282" TEXT="Execution">
<node CREATED="1383639777120" ID="ID_637843440" MODIFIED="1383639870668" TEXT="Pass symbol table with data sources so the data can be read"/>
</node>
</node>
<node CREATED="1383638494031" ID="ID_1013906206" MODIFIED="1383638502270" TEXT="Read calls">
<node CREATED="1383639599339" ID="ID_911606045" MODIFIED="1383639605561" TEXT="Annotation">
<node CREATED="1383639721743" ID="ID_1354550753" MODIFIED="1383639730206" TEXT="Open data set to determine expression type"/>
</node>
<node CREATED="1383639606100" ID="ID_687297511" MODIFIED="1383639609570" TEXT="Execution">
<node CREATED="1383639612133" ID="ID_706566603" MODIFIED="1383639635639" TEXT="Read data and add the result to the symbol table"/>
<node CREATED="1383639680784" ID="ID_42580222" MODIFIED="1383639704577" TEXT="Multiple reads of the same data should be handled as one read"/>
</node>
</node>
<node CREATED="1383639908549" ID="ID_634706132" MODIFIED="1383639914627" TEXT="Make sure data is read only once"/>
</node>
<node CREATED="1383638504041" ID="ID_449000448" MODIFIED="1383638506086" POSITION="right" TEXT="Outputs">
<node CREATED="1383638507657" ID="ID_290763998" MODIFIED="1383638906377" TEXT="Defined identifiers at global scope">
<font NAME="SansSerif" SIZE="12"/>
<node CREATED="1383638593447" ID="ID_569508855" MODIFIED="1383638609743" TEXT="Can optionally be written to datasets"/>
<node CREATED="1383638617259" ID="ID_87123442" MODIFIED="1383638660680" TEXT="Defined identifiers that aren&apos;t written to datasets and are not fed back don&apos;t need to be calculated"/>
</node>
<node CREATED="1383638524396" ID="ID_1448693392" MODIFIED="1383638527234" TEXT="Write calls"/>
</node>
<node CREATED="1383638708946" ID="ID_204829243" MODIFIED="1383638763215" POSITION="left" TEXT="A module is like a function. It as inputs and outputs. These can be determined automatically and determine the interface of the module."/>
<node CREATED="1383638793872" ID="ID_1132593331" MODIFIED="1383638796086" POSITION="left" TEXT="Misc"/>
</node>
</map>
