use Encode;
use utf8;

if (scalar(@ARGV) ne 3)
{
  print STDERR "perl LiNMT-num.pl config src-input tgt-input\n";
  exit(1);
}



$config_string = "";
open (CONFIG, "<:encoding(utf8)", $ARGV[0]) or die "Error: can not read $ARGV[0]!\n";
while (my $line = <CONFIG>)
{
  $line =~ s/[\r\n]//g;
  $config_string .= $line;
}
close(CONFIG);



open (INPUTSRC, "<:encoding(utf8)", $ARGV[1]) or die "Error: can not read $ARGV[1]!\n";
open (INPUTTGT, "<:encoding(utf8)", $ARGV[2]) or die "Error: can not read $ARGV[2]!\n";
open (OUTPUTSRC, ">:encoding(utf8)", $ARGV[1].".out") or die "Error: can not write $ARGV[1].out!\n";
open (OUTPUTTGT, ">:encoding(utf8)", $ARGV[2].".out") or die "Error: can not write $ARGV[2].out!\n";


$line_num = 0;
$find_num = 0;
while (my $line = <INPUTSRC>)
{
  ++$line_num;
  
  my $line_tgt= <INPUTTGT>;
  $line =~ s/[\r\n]//g;
  $line_tgt =~ s/[\r\n]//g;
  if ($line =~ /[[0-9$config_string]/)
  {
    ++$find_num;
    print OUTPUTSRC $line."\n";
	print OUTPUTTGT $line_tgt."\n";
    print STDERR "\rLINE=$line_num FIND=$find_num";
  }
}
print STDERR "\rLINE=$line_num FIND=$find_num\n";


close(INPUTSRC);
close(INPUTTGT);
close(OUTPUTSRC);
close(OUTPUTTGT);

