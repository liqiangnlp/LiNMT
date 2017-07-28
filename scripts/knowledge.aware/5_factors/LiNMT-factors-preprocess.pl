use strict;

if (scalar(@ARGV) ne 6)
{
  print STDERR "perl LiNMT-factors-preprocess.pl SOURCE POS NE CHK TREE OUTPUT\n";
  exit(1);
}

open (SOURCE, "<", $ARGV[0]) or die "Error: can not read $ARGV[0]!\n";
open (POS, "<", $ARGV[1]) or die "Error: can not read $ARGV[2]!\n";
open (NE, "<", $ARGV[2]) or die "Error: can not read $ARGV[3]!\n";
open (CHK, "<", $ARGV[3]) or die "Error: can not read $ARGV[4]!\n";
open (TREE, "<", $ARGV[4]) or die "Error: can not read $ARGV[5]!\n";
open (OUTPUT, ">", $ARGV[5]) or die "Error: can not write $ARGV[6]!\n";

my $error_num = 0;
my $line_num = 0;
while (my $source_line = <SOURCE>)
{
  ++$line_num;
  my $pos_line = <POS>;
  my $ne_line = <NE>;
  my $chk_line = <CHK>;
  my $tree_line = <TREE>;

  $source_line =~ s/[\r\n]//g;
  $pos_line =~ s/[\r\n]//g;
  $ne_line =~ s/[\r\n]//g;
  $chk_line =~ s/[\r\n]//g;
  $tree_line =~ s/[\r\n]//g;
  
  my @words = split /\s+/, $source_line;
  my @pos = split /\s+/, $pos_line;
  my @nes = split /\s+/, $ne_line;
  my @chks = split /\s+/, $chk_line;
  my @trees = split /\s+/, $tree_line;

  if (scalar(@words) ne scalar(@pos) or scalar(@words) ne scalar(@nes) or scalar(@words) ne scalar(@chks) or scalar(@words) ne scalar(@trees))
  {
    #print scalar(@words)." ".scalar(@pos)." ".scalar(@nes)." ".scalar(@chks)." ".scalar(@trees)."\n";
    ++$error_num;
    print OUTPUT "UNK|UNK|UNK|UNK|UNK\n"
  }
  else
  {
    my $output_line = "";
    for my $i (0..scalar(@words)-1)
    {
      $words[$i] =~ s/\|//g;
      $pos[$i] =~ s/\|//g;
      $nes[$i] =~ s/\|//g;
      $chks[$i] =~ s/\|//g;
      $trees[$i] =~ s/\|//g;
      $output_line .= " ".$words[$i]."|".$pos[$i]."|".$nes[$i]."|".$chks[$i]."|".$trees[$i];
    }
    $output_line =~ s/^\s+//g;
    $output_line =~ s/\s+$//g;
    print OUTPUT $output_line."\n";
  }

  if ($line_num % 10000 == 0)
  {
    print STDERR "\r  LINE=$line_num ERROR=$error_num";
  }
 
}
print STDERR "\r  LINE=$line_num ERROR=$error_num\n";



close (SOURCE);
close (POS);
close (NE);
close (CHK);
close (TREE);
close (OUTPUT);



